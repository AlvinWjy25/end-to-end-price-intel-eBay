with source_data as (
    select * from {{ ref('stg_ebay_listings') }}
),

volume_extraction as (
    select
        *,
        -- ============================================================
        -- Tier 1 (high): Issue Number aspect, RANGE form (e.g. "1-26",
        -- "1 to 27"). This aspect is eBay's own structured field for
        -- "which issues/volumes are included in this listing" and is
        -- consistently populated with clean ranges for lot listings
        -- (verified via diagnostic query against raw.ebay_listings).
        -- When present, this is trusted directly and title regex is
        -- skipped entirely for this item.
        -- ============================================================
        -- NOTE: negative lookahead (?!ST|ND|RD|TH) rejects patterns like
        -- "#6-1ST" (meaning "1st printing", not a volume range ending at 1).
        -- Verified via diagnostic: item "Spy Classroom SC A Light Novel #6-1ST"
        -- was previously misparsed as range 6-1.
        (regexp_match(aspect_issue_number, '^\s*(?:vol\.?\s*)?(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)(?!\s*(?:st|nd|rd|th))\s*', 'i'))[1]::float
            as tier1_issue_range_start,
        (regexp_match(aspect_issue_number, '^\s*(?:vol\.?\s*)?(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)(?!\s*(?:st|nd|rd|th))\s*', 'i'))[2]::float
            as tier1_issue_range_end,

        -- Tier 1b (high): Issue Number aspect, SINGLE value form (e.g. "24").
        -- Only relevant when the range pattern above does not match.
        case
            when aspect_issue_number ~* '^\s*\d+(\.\d+)?\s*$'
                then aspect_issue_number::float
            else null
        end as tier1b_issue_single_volume,

        -- Tier 2 (high): Book Title aspect, single "Vol. N" form.
        -- Coverage is partial (~49%) and inconsistent -- eBay sometimes
        -- truncates this field before the volume number, or populates it
        -- with only the series name. Verified via diagnostic query.
        (regexp_match(aspect_book_title, 'Vol\.?\s*(\d+(?:\.\d+)?)', 'i'))[1]::float
            as tier2_book_title_volume,

        -- Tier 3 (medium): title-based regex, single volume.
        -- Used only when Tier 1/1b/2 all yield nothing. Includes
        -- 'part/parts' alias for series like Ascendance of a Bookworm,
        -- which labels arcs/volumes as "Part N" instead of "Vol. N".
        (regexp_match(title, '(?:vol\.?|vols\.?|volume|volumes|part|parts)\s*(\d+(?:\.\d+)?)', 'i'))[1]::float
            as tier3_title_volume,

        -- Tier 3 (medium): title-based regex, '#N' shorthand form
        -- (e.g. "Spy Classroom SC A Light Novel #6-1ST"). Digit count
        -- capped at 1-2 to avoid matching SKU-style numbers (e.g.
        -- "SKU#4829102"), and negative lookahead rejects "#6-1ST" being
        -- read as "volume 61" -- the "-1ST" suffix means "1st printing",
        -- not a second number. Only ~15 items use this pattern; kept
        -- narrow on purpose.
        (regexp_match(title, '#\s*(\d{1,2})(?:\.\d+)?\b(?!\s*(?:st|nd|rd|th))', 'i'))[1]::float
            as tier3_title_hash_volume,

        -- Tier 3 (medium): title-based regex, range form. 'part/parts'
        -- alias included for the same reason as above.
        -- NOTE: multi-dash edge case (e.g. "Volumes 1-2-3-4") is a known
        -- follow-up fix, not yet applied in this pass.
        (regexp_match(title, '(?:vol\.?|vols\.?|volume|volumes|part|parts)\s*(\d+)\s*-\s*(?:vol\.?|vols\.?|volume|volumes|part|parts)?\s*(\d+)', 'i'))[1]::float
            as tier3_title_range_start,
        (regexp_match(title, '(?:vol\.?|vols\.?|volume|volumes|part|parts)\s*(\d+)\s*-\s*(?:vol\.?|vols\.?|volume|volumes|part|parts)?\s*(\d+)', 'i'))[2]::float
            as tier3_title_range_end,

        -- Tier 4 (medium): description-based regex, used ONLY when title
        -- has no volume number at all (e.g. title = "Re:Zero Full Set").
        -- Sellers are expected to spell out included volumes in the
        -- description when the title itself is generic, so this is a
        -- legitimate fallback rather than noise scraped from free text.
        (regexp_match(description, '(?:vol\.?|vols\.?|volume|volumes)\s*(\d+(?:\.\d+)?)', 'i'))[1]::float
            as tier4_desc_volume,
        (regexp_match(description, '(?:vol\.?|vols\.?|volume|volumes)\s*(\d+)\s*-\s*(?:vol\.?|vols\.?|volume|volumes)?\s*(\d+)', 'i'))[1]::float
            as tier4_desc_range_start,
        (regexp_match(description, '(?:vol\.?|vols\.?|volume|volumes)\s*(\d+)\s*-\s*(?:vol\.?|vols\.?|volume|volumes)?\s*(\d+)', 'i'))[2]::float
            as tier4_desc_range_end,

        -- Lot signal from aspects, independent of volume extraction.
        -- Cross-checks (does not override) the title-based is_boxset flag.
        case
            when aspect_unit_of_sale ~* '(lot|set|bundle)' then true
            when aspect_unit_of_sale ~* '(single|unit)' then false
            else null
        end as aspect_suggests_lot,

        case    
            when ((is_boxset is true
             or description ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+\s*-\s*\d+') and description ~* '\d+\.\d+')
            then true
            else false
        end as boxset_side_story_edition_included,

        case    
            when title ~* 'vol. \d+\.\d+'
            and title !~* '(?:vol\.?,?|vols\.?,?|volume|volumes)\s*\d+\s*-\s*\d+'
            then true
            else false
        end as standalone_side_story_edition,

        case 
            when title ~* '(exclusive|bonus|special|platinum|collector|booklet|limited|fanbook|obi|royal)'
            then true
            else false
        end as is_special_edition,

        case
            when (title ~* '(first print|first edition|1st print| 1st edition)')
            then true
            else false
        end as is_first_print,

        case
            when (description ~* '(sign|signed|autograph|autographed|直筆サイン)') or
                 (title ~* '(sign|signed|autograph|autographed|直筆サイン)')
            then true
            else false
        end as has_signature,

        case
            when (description ~* '(tapestry|standee|acrylic|figure|bonus)') or
                 (title ~* '(tapestry|standee|acrylic|figure|bonus)')
            then true
            else false
        end as has_merch,

        case 
            when (description ~* '(obi|booklet|drama cd|clear file)') or
                 (title ~* '(obi|booklet|drama cd|clear file)')
            then true
            else false
        end as has_paper_extra

    from source_data
),

count_total_bonus as (
    select
        *,
        (COALESCE(has_signature, false)::INT + 
         COALESCE(has_merch, false)::INT + 
         COALESCE(has_paper_extra, false)::INT + 
         COALESCE(is_first_print, false)::INT) AS total_bonus_count
    from volume_extraction
),

volume_resolved as (
    select
        *,
        case
            -- Logika anti-bentrok: Jika ada rentang di judul tapi aspek hanya memberi angka tunggal, 
            -- prioritaskan start range dari judul untuk mencegah anomali start > end.
            when tier3_title_range_start is not null and tier1_issue_range_start is null
            then tier3_title_range_start
            
            -- Fallback ke logika normal
            else coalesce(
                tier1_issue_range_start,
                tier1b_issue_single_volume,
                tier2_book_title_volume,
                tier3_title_range_start,
                tier3_title_volume,
                tier3_title_hash_volume,
                case when title !~* '\d' then tier4_desc_range_start end,
                case when title !~* '\d' then tier4_desc_volume end
            )
        end as volume_number,

        -- Resolve volume_number_end (end of range only; null for single volumes)
        coalesce(
            tier1_issue_range_end,
            tier3_title_range_end,
            case when title !~* '\d' then tier4_desc_range_end end
        ) as volume_number_end,

        case
            when title ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+(?:\s*,\s*\d+)+' then 'low'

            when tier1_issue_range_start is not null then 'high'
            when tier1b_issue_single_volume is not null then 'high'
            when tier2_book_title_volume is not null then 'high'
            when tier3_title_range_start is not null then 'medium'
            when tier3_title_volume is not null then 'medium'
            when tier3_title_hash_volume is not null then 'medium'
            -- when title !~* '\d' and (tier4_desc_range_start is not null or tier4_desc_volume is not null) then 'medium'
            when title !~* '\d' and (tier4_desc_range_start is not null or tier4_desc_volume is not null) then
                case
                    when is_boxset is true and tier4_desc_range_start is null then 'low'
                    else 'medium'
                end
            else 'low'
        end as volume_confidence,

        -- ============================================================
        -- title_desc_mismatch: independent QA flag. Fires whenever BOTH
        -- title and description contain a volume number/range and they
        -- disagree. Does NOT affect volume_number/volume_number_end --
        -- title remains the source of truth for extraction. Intended
        -- for manual review, not automated correction.
        -- ============================================================
        case
            when tier4_desc_volume is not null or tier4_desc_range_start is not null
            then
                case
                    -- both ranges present, but differ
                    when tier3_title_range_start is not null and tier4_desc_range_start is not null
                         and (tier3_title_range_start is distinct from tier4_desc_range_start
                              or tier3_title_range_end is distinct from tier4_desc_range_end)
                    then true
                    -- both single volumes present, but differ
                    when tier3_title_volume is not null and tier4_desc_volume is not null
                         and tier3_title_volume is distinct from tier4_desc_volume
                    then true
                    else false
                end
            else false
        end as title_desc_mismatch

    from count_total_bonus
),

metrics_calculation as (
    select 
        *,
        -- Menghitung harga per volume
        case
            -- Jika berupa boxset/rentang, bagi harga dengan jumlah buku
            when is_boxset is true and volume_number_end is not null
            then (price / NULLIF(volume_number_end - volume_number + 1, 0))
            
            -- Jika single volume, harga per volume adalah harga barang itu sendiri
            else price 
        end as price_per_volume,

        case
            -- If we have a valid range where end >= start, count is end - start + 1
            when volume_number_end is not null 
                and volume_number is not null 
                and (volume_number_end - volume_number) >= 0
            then (volume_number_end - volume_number + 1)
            
            -- If end < start (anomaly/negative delta), default to 1 (matching your np.where logic)
            when volume_number_end is not null 
                and volume_number is not null 
                and (volume_number_end - volume_number) < 0
            then 1
            
            -- If it's a single volume (no end range, but has a start volume), the count is 1
            when volume_number is not null and volume_number_end is null
            then 1
            
            -- Fallback for listings where no volume could be extracted (default to 1)
            else 1
        end as volume_count
        
    from volume_resolved
),

risk_scoring as (
    select 
        *,
        -- Text risk score with negation-aware matching
        case
            when (
                title ~* '(reprint|unbranded|bootleg|pdf|ebook|custom print|reading edition)'
                or description ~* '(reprint|not original|loose|print on demand|not an official|not official|not suitable|fan translated|fan-translated|not from original|reading edition)'
                or condition is NULL
            ) 
            and not (
                -- True Positive protection: negate the risk keywords within 4 words
                description ~* '(not|no|never|isn''t|isnt|without)\s+(\w+\s+){0,4}(reprint|unofficial|fan.?translated|bootleg)'
                or
                title ~* '(not|no|never)\s+(\w+\s+){0,4}(reprint|unofficial|bootleg)'
            )
            then 70
            else 0
        end as text_risk_score,

        case
            when (
                title ~* '(reprint|unbranded|bootleg|pdf|ebook|custom\s+print|reading\s+edition)'
                or description ~* '(reprint|not\s+original|loose|print\s+on\s+demand|not\s+an?\s+official|not\s+suitable\s+for\s+collector|not\s+(?:intended|recommended)\s+for\s+collector|fan[\s-]?translated|not\s+from\s+(?:the\s+)?original|reading\s+edition|independent\s+publisher|underground\s+publisher|not\s+published\s+by)'
                or condition is NULL
            )
            and not (
                -- True Positive protection: negate the risk keywords within 4 words
                -- (unchanged from original -- verified still correctly negates
                -- legitimate 'not a reprint/bootleg/unofficial' positive claims
                -- even with the new broader raw-match patterns above)
                description ~* '(not|no|never|isn''t|isnt|without)\s+(\w+\s+){0,4}(reprint|unofficial|fan.?translated|bootleg)'
                or
                title ~* '(not|no|never)\s+(\w+\s+){0,4}(reprint|unofficial|bootleg)'
            )
            then 70
            else 0
        end as text_risk_score_v2,

        case
            when is_boxset is true 
                and volume_number_end is not null
                and (price / NULLIF(volume_number_end - volume_number + 1, 0)) < 5.00  -- contoh threshold
            then 30

            when price < 10.00 and condition !~* 'acceptable|used' -- Allow 'Acceptable & used' condition below $8
            then 15

            else 0
        end as price_risk_score

    from metrics_calculation
)

select
    *,
    (text_risk_score_v2 + price_risk_score) as total_risk_score,

    case
        when (text_risk_score_v2 + price_risk_score) >= 50
            then 'High Risk'
        else 'Low Risk'
    end as risk_category,

    -- Look for regression model documentation to find out why
    case
        when volume_count >= 20
            and price_per_volume < 4.4
        then true
        else false
    end as is_ambigous_bulk_pricing

from risk_scoring
