# Islamic Inheritance Rules (Mirath)
# This dictionary encodes the main fixed shares and conditions for common heirs.
# More rules and edge cases can be added as needed.

inheritance_rules = {
    # Spouse rules
    'wife': {
        'share_no_children': 1/4,   # Wife gets 1/4 if deceased has no children
        'share_with_children': 1/8, # Wife gets 1/8 if deceased has children
    },
    'husband': {
        'share_no_children': 1/2,   # Husband gets 1/2 if deceased has no children
        'share_with_children': 1/4, # Husband gets 1/4 if deceased has children
    },
    # Parents
    'mother': {
        'share_no_children_or_siblings': 1/3, # Mother gets 1/3 if no children or multiple siblings
        'share_with_children_or_siblings': 1/6, # Mother gets 1/6 if deceased has children or multiple siblings
    },
    'father': {
        'share_with_children': 1/6, # Father gets 1/6 if deceased has children, plus remainder (ta'sib) if only sons
        'share_no_children': 'taasib', # Father gets remainder if no children
    },
    # Children
    'daughter': {
        'share_only_one': 1/2,      # One daughter, no sons: 1/2
        'share_two_or_more': 2/3,   # Two or more daughters, no sons: 2/3
        'with_sons': 'taasib',      # With sons: remainder, male gets double female
    },
    'son': {
        'with_daughters': 'taasib', # With daughters: remainder, male gets double female
        'only_son': 'taasib',       # Only son: all remainder
    },
    # Siblings (examples, not exhaustive)
    'full_sister': {
        'share_only_one': 1/2,      # One full sister, no brother, no children, no father
        'share_two_or_more': 2/3,   # Two or more full sisters, no brother, no children, no father
        'with_brothers': 'taasib',  # With brothers: remainder, male gets double female
    },
    'full_brother': {
        'with_sisters': 'taasib',   # With sisters: remainder, male gets double female
        'only_brother': 'taasib',   # Only brother: all remainder
    },
    # Blocking rules (examples)
    'blocking': {
        'son_blocks_siblings': True, # Presence of son blocks siblings
        'father_blocks_uncles': True, # Presence of father blocks paternal uncles
    },
    # Distribution rule
    'distribution': {
        'male_female_ratio': 2, # Males get double the share of females among children/siblings
    },
}

# This is a starting point. More detailed rules and edge cases can be added as needed. 