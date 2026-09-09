with source as (
    select * from {{ source('raw', 'ebay_listings') }}
),

cleaned as (
    select
        item_id,
        case 
            when price > 0 then price 
            else null 
        end as price,
        trim(title) as title,
        upper(trim(currency)) as currency,
        initcap(trim(condition)) as condition,
        upper(trim(seller_location)) as seller_location,
        trim(description) as description,
        localized_aspects,
        cast(created_at as timestamptz) as fetched_at,

        case
            when title ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+\s*-\s*(?:vol\.?|vols\.?|volume|volumes)?\s*\d+' 
            or title ~* '(complete set|full set|lot)'
            or title ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+(?:\s*,\s*\d+)+'
            then true
            else false
        end as is_boxset

    from source
    where item_id is not null
),

-- Tier 0: flatten localized_aspects JSONB into flat columns.
-- Avoids re-parsing the JSONB array in every downstream CTE.
-- eBay's localizedAspects is a list of {"name", "type", "value"} objects;
-- we pull out only the aspects relevant to volume extraction and lot detection.
aspects_flattened as (
    select
        item_id,
        (
            select la->>'value'
            from jsonb_array_elements(localized_aspects) as la
            where la->>'name' = 'Book Title'
            limit 1
        ) as aspect_book_title,
        (
            select la->>'value'
            from jsonb_array_elements(localized_aspects) as la
            where la->>'name' = 'Issue Number'
            limit 1
        ) as aspect_issue_number,
        (
            select la->>'value'
            from jsonb_array_elements(localized_aspects) as la
            where la->>'name' = 'Unit of Sale'
            limit 1
        ) as aspect_unit_of_sale
    from cleaned
    where localized_aspects is not null
),

simple_title_analysis as (
    select 
        item_id,
        title,
        length(title) as title_length,
        array_length(regexp_split_to_array(trim(title), '\s+'), 1) as title_word_count
    from cleaned
),

joined as (
    select
        c.*,
        a.aspect_book_title,
        a.aspect_issue_number,
        a.aspect_unit_of_sale,
        s.title_length,
        s.title_word_count
    from cleaned c
    left join aspects_flattened a using (item_id)
    left join simple_title_analysis s using (item_id)
),

normalized as (
    select
        *,
        regexp_replace(trim(title), '\s+', ' ', 'g') as normalized_title
    from joined
),

flagged as (
    select
        *,
        count(*) over (partition by normalized_title, price) as group_size
    from normalized
),

deduplicate_1 as (
    select distinct on (normalized_title, price) *
    from flagged
    order by normalized_title, price, fetched_at desc
),


deduplicated as (
    select distinct on (item_id) *
    from deduplicate_1
    order by item_id, fetched_at desc
),

light_novel_only as(
    select *
    from deduplicated
    where title !~* 'manga'
        and title !~* 'figure'
        and title !~* 'tapestry'
        and title !~* 'artbook'
        and title !~* 'poster'
        and title !~* 'blanket'
        and title !~* 'acrylic'
        and title !~* 't-shirt'
        and title !~* 'cosplay'
        and title !~* 'keychain'
        and title !~* 'sticker'
        and title !~* 'wall scroll'
        and title !~* 'phone case'
        and title !~* 'figurine'
        and title !~* 'diorama'
        and title !~* 'nendoroid'
        and title !~* 'figma'
        and title !~* ' statue'
        and title !~* ' scale'
        and title !~* 'prize'
        and title !~* 'bandai'
    order by item_id, fetched_at desc
)

select
    *
from light_novel_only