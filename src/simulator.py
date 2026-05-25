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
        prob_array = probs.values if hasattr(probs, "values") else probs

        if mask is not None:
            mask_array = mask.values if hasattr(mask, "values") else mask
            prob_array = prob_array[mask_array]

        n_total = len(prob_array)

        # targeting logic
        if top_pct is not None:
            percentile_threshold = float(np.percentile(prob_array, 100 - top_pct))
            targeted = prob_array >= percentile_threshold
        else:
            targeted = prob_array >= threshold

        target_probs = prob_array[targeted]
        n_targeted = len(target_probs)
        pct_targeted = n_targeted / n_total if n_total else 0

        # baseline churn in targeted group
        churners = int(target_probs.sum())
        churn_rate = float(target_probs.mean()) if n_targeted else 0

        # intervention effect
        saves = int(churners * campaign.lift * campaign.response_rate)
        saves_revenue = saves * self.ltv

        # costs
        contact_cost = n_targeted * campaign.cost_per_contact
        discount_cost = n_targeted * campaign.response_rate * campaign.discount
        total_cost = contact_cost + discount_cost

        # roi
        incremental_revenue = saves_revenue - total_cost
        roi = incremental_revenue / total_cost if total_cost else 0
        cost_per_save = total_cost / saves if saves else 1e9

        # break-even
        breakeven_denominator = churners * campaign.response_rate * self.ltv
        breakeven_lift = total_cost / breakeven_denominator if breakeven_denominator else 1e9

        return Result(
            campaign=campaign.name,
            n_total=n_total,
            n_targeted=n_targeted,
            pct_targeted=pct_targeted,
            churners_baseline=churners,
            churn_rate_target=churn_rate,
            saves=saves,
            saves_rev=saves_revenue,
            contact_cost=contact_cost,
            discount_cost=discount_cost,
            total_cost=total_cost,
            inc_rev=incremental_revenue,
            roi=roi,
            cps=cost_per_save,
            be_lift=breakeven_lift,
        )

    def compare(self, probs, scenarios: list[dict]) -> pd.DataFrame:
        """Compare multiple scenarios at once."""
        rows = []
        for scenario in scenarios:
            campaign = CampaignParams(
                name=scenario.get("name", "Scenario"),
                cost_per_contact=scenario.get("cost", 5),
                discount=scenario.get("discount", 10),
                lift=scenario.get("lift", 0.2),
                response_rate=scenario.get("response", 0.3),
            )
            result = self.run(
                probs, campaign, threshold=scenario.get("threshold", 0.5),
                top_pct=scenario.get("top_pct"),
            )
            rows.append(result.to_dict())
        return pd.DataFrame(rows)

    def optimize(self, probs, campaign=None, thresholds=None) -> pd.DataFrame:
        """Find best threshold."""
        campaign = campaign or CampaignParams()
        thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        rows = []
        for threshold_value in thresholds:
            result = self.run(probs, campaign, threshold=threshold_value)
            rows.append(
                {
                    "thresh": threshold_value,
                    "pct": result.pct_targeted,
                    "targeted": result.n_targeted,
                    "saves": result.saves,
                    "cost": result.total_cost,
                    "inc_rev": result.inc_rev,
                    "roi": result.roi,
                    "cps": result.cps,
                }
            )

        results_df = pd.DataFrame(rows)
        # mark optimal (max roi with positive revenue)
        profitable = results_df[results_df["inc_rev"] > 0]
        if len(profitable) > 0:
            best_idx = profitable["roi"].idxmax()
            results_df["best"] = results_df.index == best_idx
        else:
            results_df["best"] = False

        return results_df

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
                campaign = CampaignParams(
                    cost_per_contact=base.cost_per_contact,
                    discount=base.discount,
                    lift=lift,
                    response_rate=base.response_rate,
                )
                result = self.run(probs, campaign, threshold)
                rows.append(
                    {"lift": lift, "roi": result.roi, "inc_rev": result.inc_rev,
                     "saves": result.saves}
                )
            results["lift"] = pd.DataFrame(rows)

        # cost sensitivity
        if "cost" in ranges:
            rows = []
            for cost in ranges["cost"]:
                campaign = CampaignParams(
                    cost_per_contact=cost,
                    discount=base.discount,
                    lift=base.lift,
                    response_rate=base.response_rate,
                )
                result = self.run(probs, campaign, threshold)
                rows.append(
                    {"cost": cost, "roi": result.roi, "inc_rev": result.inc_rev,
                     "total": result.total_cost}
                )
            results["cost"] = pd.DataFrame(rows)

        # ltv sensitivity
        if "ltv" in ranges:
            rows = []
            original_ltv = self.ltv
            for ltv in ranges["ltv"]:
                self.ltv = ltv
                result = self.run(probs, base, threshold)
                rows.append(
                    {"ltv": ltv, "roi": result.roi, "inc_rev": result.inc_rev,
                     "saves_rev": result.saves_rev}
                )
            self.ltv = original_ltv
            results["ltv"] = pd.DataFrame(rows)

        return results

    def targeting_list(self, ids, probs, segments=None, threshold=0.5, top_n=None) -> pd.DataFrame:
        """Export list of customers to target."""
        targets = pd.DataFrame(
            {
                "id": ids.values if hasattr(ids, "values") else ids,
                "risk": probs.values if hasattr(probs, "values") else probs,
            }
        )

        if segments is not None:
            targets["seg"] = segments.values if hasattr(segments, "values") else segments

        targets = targets[targets["risk"] >= threshold]
        targets = targets.sort_values("risk", ascending=False)

        if top_n:
            targets = targets.head(top_n)

        # priority tiers
        targets["priority"] = pd.qcut(
            targets["risk"].rank(method="first"), q=3, labels=["Med", "Hi", "Critical"]
        )
        targets["exp_value"] = targets["risk"] * self.ltv

        return targets.reset_index(drop=True)


def quick_roi(n, churn_rate, ltv, cost=5, lift=0.2, response=0.3) -> dict:
    """Back of envelope calculation."""
    churners = n * churn_rate
    saves = churners * lift * response
    total_cost = n * cost
    revenue = saves * ltv
    incremental_revenue = revenue - total_cost
    roi = incremental_revenue / total_cost if total_cost else 0

    return {
        "targeted": n,
        "churners": int(churners),
        "saves": int(saves),
        "cost": f"${total_cost:,.0f}",
        "rev": f"${revenue:,.0f}",
        "inc_rev": f"${incremental_revenue:,.0f}",
        "roi": f"{roi:.0%}",
        "profitable": incremental_revenue > 0,
    }
