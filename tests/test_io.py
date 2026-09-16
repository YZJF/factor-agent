from factor_agent.io import case_from_dict, load_jsonl


def test_alias_id_spec():
    case = case_from_dict(
        {
            "id": "x",
            "report": "中证500 20日反转，行业中性。",
            "spec": {
                "name": "a",
                "universe": "CSI500",
                "frequency": "daily",
                "neutralization": "industry",
                "rebalance": "weekly",
                "expr": "TS_PCTCHANGE($close, 20)",
                "windows": [20],
            },
        }
    )
    assert case.case_id == "x"
    assert case.gold.universe == "CSI500"


def test_load_demo_gold():
    cases = load_jsonl("data/gold/gold.jsonl")
    assert len(cases) >= 3
    assert cases[0].gold.expr
