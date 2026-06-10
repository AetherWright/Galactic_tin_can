"""Static flavour events available to every nation."""

EVENTS = {
    "plague": [
        {
            "choice": "Quarantine affected cities",
            "effects": {"plague": -0.02, "stability": -5},
        },
        {"choice": "Ignore the plague", "effects": {"plague": 0.02}},
        {"choice": "Seek foreign aid", "effects": {"economy": 10, "plague": -0.01}},
    ],
    "gold rush": [
        {"choice": "Nationalize resources", "effects": {"economy": 20, "stability": 3}},
        {"choice": "Open to investors", "effects": {"economy": 30, "stability": -3}},
    ],
    "tech breakthrough": [
        {"choice": "Invest heavily in research", "effects": {"science": 5, "economy": -10}},
        {"choice": "Patent and sell", "effects": {"economy": 15, "stability": 2}},
    ],
    "political scandal": [
        {"choice": "Cover it up", "effects": {"stability": -10}},
        {"choice": "Public inquiry", "effects": {"stability": -5, "economy": -5}},
        {"choice": "Resign gracefully", "effects": {"stability": 3, "economy": -3}},
     ]
}

