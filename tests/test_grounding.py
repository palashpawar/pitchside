from pitchside.agents.grounding import check

ROSTER = ["Nico Vargas", "Sami Haddad", "Viktor Hallberg"]
TEAMS = ["Kingsmoor FC", "Harbourside Albion"]
FACTS = {"scorer": "Nico Vargas", "xg": 0.348, "move_passes": 6, "score": "2-1", "team": "Kingsmoor FC",
         "home_team": "Kingsmoor FC", "away_team": "Harbourside Albion", "pass_accuracy": 0.87}


def test_grounded_text_passes():
    assert check("Nico Vargas finishes a 0.35 xG chance after 6 passes: 2-1 Kingsmoor FC.", FACTS, ROSTER, TEAMS) == []


def test_percentages_of_proportions_are_allowed():
    assert check("They are passing at 87% accuracy.", FACTS, ROSTER, TEAMS) == []


def test_invented_number_is_caught():
    issues = check("Vargas has now scored 14 goals this season.", FACTS, ROSTER, TEAMS)
    assert any("14" in i for i in issues)


def test_player_not_in_moment_is_caught():
    issues = check("Hallberg was nowhere near it.", FACTS, ROSTER, TEAMS)
    assert any("Viktor Hallberg" in i for i in issues)


def test_spanish_decimal_comma():
    assert check("Una ocasión de 0,35 xG.", FACTS, ROSTER, TEAMS) == []


def test_ordinals_are_checked():
    assert any("27" in i for i in check("His 27th goal.", FACTS, ROSTER, TEAMS))
