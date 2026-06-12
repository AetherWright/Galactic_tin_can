"""Procedural event generation keyed on a nation's current situation."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..planets import PLANETS


class ProceduralEventGenerator:
    """Create simple events based on a nation's state."""

    def generate(self, nation, nations: Dict) -> Tuple[str, List[Dict]]:
        """Generate contextually appropriate events based on actual nation state."""
        
        # Build situation profile
        at_war = len(nation.at_war) > 0
        neighbors_at_war = any(
            len(n.at_war) > 0 for nid, n in nations.items() 
            if nid in nation.relations
        )
        high_tech = nation.technology.science > 30
        struggling = nation.economy_linear < 100
        large = len(nation.cities) > 5
        unstable = nation.stability < 40
        prosperous = nation.economy_linear > 300 and nation.stability > 60
        has_nukes = "Nuclear Weapons" in nation.tech_tree.unlocked
        plague_nearby = any(
            PLANETS.get(c.planet) and PLANETS[c.planet].plague_level > 0.1
            for c in nation.cities
        )
        
        # Weighted event pool based on actual situation
        pool = []
   
        if at_war:
            pool += [
                ("supply shortage", [
                    {"choice": "Ration supplies", "effects": {"economy": -15, "stability": -3}},
                    {"choice": "Emergency production", "effects": {"economy": -25, "stability": 5}},
                ]),
                ("war hero emerges", [
                    {"choice": "Promote publicly", "effects": {"stability": 10, "military": 5}},
                    {"choice": "Keep quiet", "effects": {"military": 3}},
                ]),
                ("desertion crisis", [
                    {"choice": "Execute deserters", "effects": {"stability": -10, "military": 5}},
                    {"choice": "Offer amnesty", "effects": {"stability": 5, "military": -5}},
                ]),
            ] * 3  # weight war events heavily when at war
    
        if plague_nearby or any(
            getattr(PLANETS.get(c.planet), 'plague_level', 0) > 0 
            for c in nation.cities
        ):
            pool += [
                ("epidemic spreading", [
                    {"choice": "Strict quarantine", "effects": {"stability": -8, "economy": -10}},
                    {"choice": "Develop treatment", "effects": {"economy": -20, "science": 3}},
                    {"choice": "Ignore", "effects": {"plague": 0.05}},
                ]),
            ] * 2
    
        if high_tech and not at_war:
            pool += [
                ("research breakthrough", [
                    {"choice": "Publish findings", "effects": {"science": 8, "stability": 3}},
                    {"choice": "Classify research", "effects": {"science": 4, "military": 5}},
                    {"choice": "Sell to industry", "effects": {"economy": 20, "science": 2}},
                ]),
                ("automation displacement", [
                    {"choice": "Retrain workers", "effects": {"economy": -15, "stability": 5}},
                    {"choice": "Let market adjust", "effects": {"economy": 10, "stability": -8}},
                    {"choice": "Regulate automation", "effects": {"economy": -5, "stability": 3}},
                ]),
            ]
        
        if has_nukes:
            pool += [
                ("proliferation pressure", [
                    {"choice": "Sign treaty", "effects": {"stability": 5, "military": -3}},
                    {"choice": "Refuse", "effects": {"stability": -5, "military": 5}},
                    {"choice": "Share technology", "effects": {"economy": 20, "stability": -8}},
                ]),
            ]
    
        if large and not at_war:
            pool += [
                ("administrative reform", [
                    {"choice": "Centralize", "effects": {"stability": -5, "economy": 10}},
                    {"choice": "Decentralize", "effects": {"stability": 8, "economy": -5}},
                    {"choice": "Status quo", "effects": {}},
                ]),
                ("separatist movement", [
                    {"choice": "Negotiate autonomy", "effects": {"stability": -3, "economy": -5}},
                    {"choice": "Crack down", "effects": {"stability": -12, "military": 3}},
                    {"choice": "Grant independence", "effects": {"stability": 5, "economy": -15}},
                ]),
            ]
        
        if prosperous:
            pool += [
                ("golden age", [
                    {"choice": "Invest in culture", "effects": {"stability": 12, "economy": -5}},
                    {"choice": "Expand military", "effects": {"military": 10, "stability": 3}},
                    {"choice": "Fund exploration", "effects": {"economy": -10, "science": 5}},
                ]),
                ("trade boom", [
                    {"choice": "Open borders", "effects": {"economy": 25, "stability": 5}},
                    {"choice": "Protect industry", "effects": {"economy": 10, "stability": 3}},
                ]),
            ] * 2
    
        if unstable:
            pool += [
                ("reform movement", [
                    {"choice": "Embrace reform", "effects": {"stability": 10, "economy": -10}},
                    {"choice": "Suppress", "effects": {"stability": -10, "military": 3}},
                    {"choice": "Compromise", "effects": {"stability": 5, "economy": -5}},
                ]),
                ("coup attempt", [
                    {"choice": "Crush it", "effects": {"stability": -5, "military": 5}},
                    {"choice": "Negotiate", "effects": {"stability": 3, "economy": -5}},
                    {"choice": "Flee", "effects": {"stability": -20}},
                ]),
            ] * 2
        
        if struggling:
            pool += [
                ("debt crisis", [
                    {"choice": "Austerity", "effects": {"economy": 15, "stability": -10}},
                    {"choice": "Default", "effects": {"economy": -10, "stability": -5}},
                    {"choice": "Foreign aid", "effects": {"economy": 10, "stability": 3}},
                ]),
                ("food shortage", [
                    {"choice": "Import food", "effects": {"economy": -15, "stability": 3}},
                    {"choice": "Ration", "effects": {"economy": -5, "stability": -8}},
                    {"choice": "Ignore", "effects": {"population": -1000, "stability": -10}},
                ]),
            ] * 2
    
        if neighbors_at_war and not at_war:
            pool += [
                ("refugee crisis", [
                    {"choice": "Accept refugees", "effects": {"population": 5000, "stability": -5}},
                    {"choice": "Close borders", "effects": {"stability": -3, "economy": -5}},
                    {"choice": "Exploit situation", "effects": {"economy": 15, "stability": -8}},
                ]),
                ("war profiteering opportunity", [
                    {"choice": "Sell arms", "effects": {"economy": 25, "stability": -5}},
                    {"choice": "Stay neutral", "effects": {"stability": 3}},
                    {"choice": "Mediate", "effects": {"stability": 5, "economy": -5}},
                ]),
            ]
    
            # Always have some baseline events as fallback
            pool += [
                ("census results", [
                    {"choice": "Publish data", "effects": {"stability": 2}},
                    {"choice": "Classify data", "effects": {"stability": -1}},
            ]),
            ("infrastructure aging", [
                {"choice": "Emergency repairs", "effects": {"economy": -10, "stability": 3}},
                {"choice": "Defer maintenance", "effects": {"stability": -3}},
            ]),
            ]
    
        if not pool:
            # Absolute fallback
            return "civil unrest", [
                {"choice": "Promise reforms", "effects": {"stability": 3}},
                {"choice": "Crack down", "effects": {"stability": -5}},
            ]
    
        name, choices = random.choice(pool)
        return name, choices

