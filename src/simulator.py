# ROI simulator for churn interventions
# estimates business impact of retention campaigns

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CampaignParams:
    """Settings for a retention campaign."""

    name: str = "Default"
    cost_per_contact: float = 5.0
    discount: float = 10.0
    lift: float = 0.20  # expected churn reduction
    response_rate: float = 0.30


@dataclass
class Result:
    """Simulation output."""

    campaign: str
    n_total: int
    n_targeted: int
    pct_targeted: float
    churners_baseline: int
    churn_rate_target: float
    saves: int
    saves_rev: float
    contact_cost: float
    discount_cost: float
    total_cost: float
    inc_rev: float
    roi: float
    cps: float  # cost per save
    be_lift: float  # break-even lift

    def to_dict(self) -> dict:
        return {
            "campaign": self.campaign,
            "targeted": self.n_targeted,
            "pct": f"{self.pct_targeted:.1%}",
            "saves": self.saves,
            "cost": "$" + str(int(self.total_cost)),
            "inc_rev": f"${self.inc_rev:,.0f}",
            "roi": f"{self.roi:.0%}",
            "cost_per_save": f"${self.cps:.2f}",
        }

    def summary(self) -> str:
        return f"""
Campaign: {self.campaign}
================================================================================

Targeting: {self.n_targeted:,} of {self.n_total:,} ({self.pct_targeted:.1%})

Baseline: {self.churners_baseline:,} expected churners ({self.churn_rate_target:.1%} rate)

Intervention:
  Expected saves: {self.saves:,}
  Revenue from saves: ${self.saves_rev:,.0f}

Costs:
  Contact: ${self.contact_cost:,.0f}
  Discounts: ${self.discount_cost:,.0f}
  Total: ${self.total_cost:,.0f}

ROI: {self.roi:.0%}
Cost per save: ${self.cps:.2f}
Break-even lift: {self.be_lift:.1%}
"""


class InterventionSimulator:
    """Figure out if a retention campaign makes financial sense.

    usage:
        sim = InterventionSimulator(ltv=100)
        result = sim.run(probs, threshold=0.5)
        print(result.summary())
    """

    def __init__(self, ltv: float = 100.0):
        self.ltv = ltv

    def run(self, probs, campaign=None, threshold=0.5, top_pct=None, mask=None) -> Result:
        """Run the simulation.

        probs: churn probabilities
        campaign: CampaignParams or None for defaults
        threshold: target everyone above this risk
        top_pct: or target top N% by risk (overrides threshold)
        mask: boolean series to filter to subset
        """
        campaign = campaign or CampaignParams()
        p = probs.values if hasattr(probs, "values") else probs

        if mask is not None:
            m = mask.values if hasattr(mask, "values") else mask
            p = p[m]

        n_total = len(p)

        # targeting logic
        if top_pct is not None:
            thresh = float(np.percentile(p, 100 - top_pct))
            targeted = p >= thresh
        else:
            targeted = p >= threshold

        target_probs = p[targeted]
        n_targeted = len(target_probs)
        pct = n_targeted / n_total if n_total else 0

        # baseline churn in targeted group
        churners = int(target_probs.sum())
        churn_rate = float(target_probs.mean()) if n_targeted else 0

        # intervention effect
        saves = int(churners * campaign.lift * campaign.response_rate)
        saves_rev = saves * self.ltv

        # costs
        contact_cost = n_targeted * campaign.cost_per_contact
        discount_cost = n_targeted * campaign.response_rate * campaign.discount
        total = contact_cost + discount_cost

        # roi
        inc = saves_rev - total
        roi = inc / total if total else 0
        cps = total / saves if saves else 1e9

        # break-even
        denom = churners * campaign.response_rate * self.ltv
        be = total / denom if denom else 1e9

        return Result(
            campaign=campaign.name,
            n_total=n_total,
            n_targeted=n_targeted,
            pct_targeted=pct,
            churners_baseline=churners,
            churn_rate_target=churn_rate,
            saves=saves,
            saves_rev=saves_rev,
            contact_cost=contact_cost,
            discount_cost=discount_cost,
            total_cost=total,
            inc_rev=inc,
            roi=roi,
            cps=cps,
            be_lift=be,
        )

    def compare(self, probs, scenarios: list[dict]) -> pd.DataFrame:
        """Compare multiple scenarios at once."""
        rows = []
        for s in scenarios:
            cp = CampaignParams(
                name=s.get("name", "Scenario"),
                cost_per_contact=s.get("cost", 5),
                discount=s.get("discount", 10),
                lift=s.get("lift", 0.2),
                response_rate=s.get("response", 0.3),
            )
            r = self.run(probs, cp, threshold=s.get("threshold", 0.5), top_pct=s.get("top_pct"))
            rows.append(r.to_dict())
        return pd.DataFrame(rows)

    def optimize(self, probs, campaign=None, thresholds=None) -> pd.DataFrame:
        """Find best threshold."""
        campaign = campaign or CampaignParams()
        thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        out = []
        for t in thresholds:
            r = self.run(probs, campaign, threshold=t)
            out.append(
                {
                    "thresh": t,
                    "pct": r.pct_targeted,
                    "targeted": r.n_targeted,
                    "saves": r.saves,
                    "cost": r.total_cost,
                    "inc_rev": r.inc_rev,
                    "roi": r.roi,
                    "cps": r.cps,
                }
            )

        df = pd.DataFrame(out)
        # mark optimal (max roi with positive revenue)
        pos = df[df["inc_rev"] > 0]
        if len(pos) > 0:
            best = pos["roi"].idxmax()
            df["best"] = df.index == best
        else:
            df["best"] = False

        return df

    def sensitivity(self, probs, base=None, threshold=0.5, ranges=None) -> dict[str, pd.DataFrame]:
        """How sensitive is ROI to each parameter?"""
        base = base or CampaignParams()
        ranges = ranges or {
            "lift": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
            "cost": [1, 2, 5, 10, 15, 20],
            "ltv": [50, 75, 100, 150, 200, 300],
        }

        results = {}

        # lift sensitivity
        if "lift" in ranges:
            rows = []
            for lift in ranges["lift"]:
                cp = CampaignParams(
                    cost_per_contact=base.cost_per_contact,
                    discount=base.discount,
                    lift=lift,
                    response_rate=base.response_rate,
                )
                r = self.run(probs, cp, threshold)
                rows.append({"lift": lift, "roi": r.roi, "inc_rev": r.inc_rev, "saves": r.saves})
            results["lift"] = pd.DataFrame(rows)

        # cost sensitivity
        if "cost" in ranges:
            rows = []
            for cost in ranges["cost"]:
                cp = CampaignParams(
                    cost_per_contact=cost,
                    discount=base.discount,
                    lift=base.lift,
                    response_rate=base.response_rate,
                )
                r = self.run(probs, cp, threshold)
                rows.append(
                    {"cost": cost, "roi": r.roi, "inc_rev": r.inc_rev, "total": r.total_cost}
                )
            results["cost"] = pd.DataFrame(rows)

        # ltv sensitivity
        if "ltv" in ranges:
            rows = []
            orig = self.ltv
            for ltv in ranges["ltv"]:
                self.ltv = ltv
                r = self.run(probs, base, threshold)
                rows.append(
                    {"ltv": ltv, "roi": r.roi, "inc_rev": r.inc_rev, "saves_rev": r.saves_rev}
                )
            self.ltv = orig
            results["ltv"] = pd.DataFrame(rows)

        return results

    def targeting_list(self, ids, probs, segments=None, threshold=0.5, top_n=None) -> pd.DataFrame:
        """Export list of customers to target."""
        df = pd.DataFrame(
            {
                "id": ids.values if hasattr(ids, "values") else ids,
                "risk": probs.values if hasattr(probs, "values") else probs,
            }
        )

        if segments is not None:
            df["seg"] = segments.values if hasattr(segments, "values") else segments

        df = df[df["risk"] >= threshold]
        df = df.sort_values("risk", ascending=False)

        if top_n:
            df = df.head(top_n)

        # priority tiers
        df["priority"] = pd.qcut(
            df["risk"].rank(method="first"), q=3, labels=["Med", "Hi", "Critical"]
        )
        df["exp_value"] = df["risk"] * self.ltv

        return df.reset_index(drop=True)


def quick_roi(n, churn_rate, ltv, cost=5, lift=0.2, response=0.3) -> dict:
    """Back of envelope calculation."""
    churners = n * churn_rate
    saves = churners * lift * response
    total_cost = n * cost
    rev = saves * ltv
    inc = rev - total_cost
    roi = inc / total_cost if total_cost else 0

    return {
        "targeted": n,
        "churners": int(churners),
        "saves": int(saves),
        "cost": f"${total_cost:,.0f}",
        "rev": f"${rev:,.0f}",
        "inc_rev": f"${inc:,.0f}",
        "roi": f"{roi:.0%}",
        "profitable": inc > 0,
    }
