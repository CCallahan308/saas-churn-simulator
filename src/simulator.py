"""
Intervention simulator for churn reduction campaigns.

Calculates ROI and business impact of targeting at-risk customers
with retention campaigns based on model predictions.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CampaignParams:
    """
    Parameters for a retention campaign.
    
    Attributes:
        name: Campaign name
        cost_per_contact: Cost to reach each customer ($)
        discount_value: Value of discount/offer provided ($)
        expected_lift: Expected reduction in churn rate (0-1)
        success_rate: Rate of customers who respond to contact (0-1)
    """
    name: str = "Default Campaign"
    cost_per_contact: float = 5.0
    discount_value: float = 10.0
    expected_lift: float = 0.20  # 20% reduction in churn
    success_rate: float = 0.30  # 30% of contacted respond


@dataclass
class SimulationResult:
    """Results from an intervention simulation."""
    
    campaign_name: str
    total_customers: int
    targeted_customers: int
    targeting_pct: float
    
    # Without intervention
    baseline_churners: int
    baseline_churn_rate: float
    
    # With intervention
    expected_saves: int
    expected_saves_revenue: float
    
    # Costs
    contact_cost: float
    discount_cost: float
    total_cost: float
    
    # ROI metrics
    incremental_revenue: float
    roi: float
    cost_per_save: float
    break_even_lift: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "campaign_name": self.campaign_name,
            "targeted_customers": self.targeted_customers,
            "targeting_pct": f"{self.targeting_pct:.1%}",
            "baseline_churners": self.baseline_churners,
            "expected_saves": self.expected_saves,
            "total_cost": f"${self.total_cost:,.0f}",
            "incremental_revenue": f"${self.incremental_revenue:,.0f}",
            "roi": f"{self.roi:.0%}",
            "cost_per_save": f"${self.cost_per_save:.2f}",
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Campaign: {self.campaign_name}
================================================================================

TARGETING
---------
Total customer base: {self.total_customers:,}
Targeted customers: {self.targeted_customers:,} ({self.targeting_pct:.1%})

BASELINE (No Intervention)
--------------------------
Expected churners: {self.baseline_churners:,}
Churn rate in target group: {self.baseline_churn_rate:.1%}

INTERVENTION IMPACT
-------------------
Expected customers saved: {self.expected_saves:,}
Revenue from saves: ${self.expected_saves_revenue:,.0f}

COSTS
-----
Contact cost: ${self.contact_cost:,.0f}
Discount cost: ${self.discount_cost:,.0f}
Total cost: ${self.total_cost:,.0f}

ROI ANALYSIS
------------
Incremental revenue: ${self.incremental_revenue:,.0f}
ROI: {self.roi:.0%}
Cost per save: ${self.cost_per_save:.2f}
Break-even lift required: {self.break_even_lift:.1%}
"""


class InterventionSimulator:
    """
    Simulate business impact of churn intervention campaigns.
    
    Takes model predictions and campaign parameters to estimate:
    - Number of customers to target
    - Expected saves (churners converted to retained)
    - Campaign costs
    - ROI and incremental revenue
    
    Example:
        >>> simulator = InterventionSimulator(avg_ltv=100)
        >>> result = simulator.simulate(
        ...     churn_proba=predictions,
        ...     campaign=CampaignParams(cost_per_contact=5, expected_lift=0.2),
        ...     risk_threshold=0.5
        ... )
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        avg_ltv: float = 100.0,
        time_horizon_months: int = 12,
    ):
        """
        Initialize simulator.
        
        Args:
            avg_ltv: Average customer lifetime value ($)
            time_horizon_months: Time horizon for LTV calculation
        """
        self.avg_ltv = avg_ltv
        self.time_horizon_months = time_horizon_months
        
    def simulate(
        self,
        churn_proba: pd.Series,
        campaign: Optional[CampaignParams] = None,
        risk_threshold: float = 0.5,
        top_n_pct: Optional[float] = None,
        segment_mask: Optional[pd.Series] = None,
    ) -> SimulationResult:
        """
        Run intervention simulation.
        
        Args:
            churn_proba: Series of churn probabilities per customer
            campaign: Campaign parameters (uses defaults if None)
            risk_threshold: Minimum churn probability to target
            top_n_pct: Target top N% by risk (overrides threshold)
            segment_mask: Boolean mask for segment filtering
            
        Returns:
            SimulationResult with all metrics
        """
        campaign = campaign or CampaignParams()
        proba = churn_proba.values if hasattr(churn_proba, "values") else churn_proba
        
        # Apply segment filter if provided
        if segment_mask is not None:
            mask = segment_mask.values if hasattr(segment_mask, "values") else segment_mask
            proba = proba[mask]
        
        total_customers = len(proba)
        
        # Determine targeting rule
        if top_n_pct is not None:
            # Target top N% by risk
            threshold = np.percentile(proba, 100 - top_n_pct)
            targeted_mask = proba >= threshold
        else:
            # Target by risk threshold
            targeted_mask = proba >= risk_threshold
        
        # Calculate metrics for targeted group
        targeted_proba = proba[targeted_mask]
        targeted_customers = len(targeted_proba)
        targeting_pct = targeted_customers / total_customers if total_customers > 0 else 0
        
        # Baseline (no intervention)
        baseline_churners = int(targeted_proba.sum())
        baseline_churn_rate = targeted_proba.mean() if len(targeted_proba) > 0 else 0
        
        # With intervention
        # Expected saves = churners * lift * success_rate
        expected_saves = int(
            baseline_churners * campaign.expected_lift * campaign.success_rate
        )
        expected_saves_revenue = expected_saves * self.avg_ltv
        
        # Costs
        contact_cost = targeted_customers * campaign.cost_per_contact
        # Discount cost only for those who respond
        discount_cost = targeted_customers * campaign.success_rate * campaign.discount_value
        total_cost = contact_cost + discount_cost
        
        # ROI
        incremental_revenue = expected_saves_revenue - total_cost
        roi = (incremental_revenue / total_cost) if total_cost > 0 else 0
        cost_per_save = total_cost / expected_saves if expected_saves > 0 else float("inf")
        
        # Break-even lift: what lift would we need to break even?
        # Revenue at break-even = Cost
        # saves * LTV = Cost
        # churners * lift * success_rate * LTV = Cost
        # lift = Cost / (churners * success_rate * LTV)
        break_even_lift = (
            total_cost / (baseline_churners * campaign.success_rate * self.avg_ltv)
            if baseline_churners > 0 and campaign.success_rate > 0 and self.avg_ltv > 0
            else float("inf")
        )
        
        return SimulationResult(
            campaign_name=campaign.name,
            total_customers=total_customers,
            targeted_customers=targeted_customers,
            targeting_pct=targeting_pct,
            baseline_churners=baseline_churners,
            baseline_churn_rate=baseline_churn_rate,
            expected_saves=expected_saves,
            expected_saves_revenue=expected_saves_revenue,
            contact_cost=contact_cost,
            discount_cost=discount_cost,
            total_cost=total_cost,
            incremental_revenue=incremental_revenue,
            roi=roi,
            cost_per_save=cost_per_save,
            break_even_lift=break_even_lift,
        )
    
    def compare_scenarios(
        self,
        churn_proba: pd.Series,
        scenarios: List[Dict],
    ) -> pd.DataFrame:
        """
        Compare multiple intervention scenarios.
        
        Args:
            churn_proba: Churn probabilities
            scenarios: List of scenario configs with keys:
                       - name, risk_threshold or top_n_pct, campaign params
                       
        Returns:
            DataFrame comparing scenarios
        """
        results = []
        
        for scenario in scenarios:
            name = scenario.get("name", "Scenario")
            threshold = scenario.get("risk_threshold", 0.5)
            top_n = scenario.get("top_n_pct")
            
            campaign = CampaignParams(
                name=name,
                cost_per_contact=scenario.get("cost_per_contact", 5.0),
                discount_value=scenario.get("discount_value", 10.0),
                expected_lift=scenario.get("expected_lift", 0.20),
                success_rate=scenario.get("success_rate", 0.30),
            )
            
            result = self.simulate(
                churn_proba=churn_proba,
                campaign=campaign,
                risk_threshold=threshold,
                top_n_pct=top_n,
            )
            
            results.append(result.to_dict())
        
        return pd.DataFrame(results)
    
    def optimize_threshold(
        self,
        churn_proba: pd.Series,
        campaign: Optional[CampaignParams] = None,
        thresholds: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        Find optimal risk threshold by testing multiple values.
        
        Args:
            churn_proba: Churn probabilities
            campaign: Campaign parameters
            thresholds: Thresholds to test (defaults to 0.1 to 0.9)
            
        Returns:
            DataFrame with metrics for each threshold
        """
        campaign = campaign or CampaignParams()
        thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = []
        
        for threshold in thresholds:
            result = self.simulate(
                churn_proba=churn_proba,
                campaign=campaign,
                risk_threshold=threshold,
            )
            
            results.append({
                "threshold": threshold,
                "targeted_pct": result.targeting_pct,
                "targeted_customers": result.targeted_customers,
                "expected_saves": result.expected_saves,
                "total_cost": result.total_cost,
                "incremental_revenue": result.incremental_revenue,
                "roi": result.roi,
                "cost_per_save": result.cost_per_save,
            })
        
        df = pd.DataFrame(results)
        
        # Find optimal threshold (max ROI with positive incremental revenue)
        positive_roi = df[df["incremental_revenue"] > 0]
        if len(positive_roi) > 0:
            optimal_idx = positive_roi["roi"].idxmax()
            df["is_optimal"] = df.index == optimal_idx
        else:
            df["is_optimal"] = False
        
        return df
    
    def sensitivity_analysis(
        self,
        churn_proba: pd.Series,
        base_campaign: Optional[CampaignParams] = None,
        risk_threshold: float = 0.5,
        param_ranges: Optional[Dict] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze sensitivity to campaign parameters.
        
        Args:
            churn_proba: Churn probabilities
            base_campaign: Base campaign to vary
            risk_threshold: Risk threshold to use
            param_ranges: Ranges for each parameter to test
            
        Returns:
            Dictionary of DataFrames, one per parameter
        """
        base = base_campaign or CampaignParams()
        
        param_ranges = param_ranges or {
            "expected_lift": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
            "cost_per_contact": [1, 2, 5, 10, 15, 20],
            "avg_ltv": [50, 75, 100, 150, 200, 300],
        }
        
        sensitivity = {}
        
        # Sensitivity to expected lift
        if "expected_lift" in param_ranges:
            results = []
            for lift in param_ranges["expected_lift"]:
                campaign = CampaignParams(
                    name=f"Lift={lift:.0%}",
                    cost_per_contact=base.cost_per_contact,
                    discount_value=base.discount_value,
                    expected_lift=lift,
                    success_rate=base.success_rate,
                )
                result = self.simulate(churn_proba, campaign, risk_threshold)
                results.append({
                    "expected_lift": lift,
                    "roi": result.roi,
                    "incremental_revenue": result.incremental_revenue,
                    "expected_saves": result.expected_saves,
                })
            sensitivity["expected_lift"] = pd.DataFrame(results)
        
        # Sensitivity to cost per contact
        if "cost_per_contact" in param_ranges:
            results = []
            for cost in param_ranges["cost_per_contact"]:
                campaign = CampaignParams(
                    name=f"Cost=${cost}",
                    cost_per_contact=cost,
                    discount_value=base.discount_value,
                    expected_lift=base.expected_lift,
                    success_rate=base.success_rate,
                )
                result = self.simulate(churn_proba, campaign, risk_threshold)
                results.append({
                    "cost_per_contact": cost,
                    "roi": result.roi,
                    "incremental_revenue": result.incremental_revenue,
                    "total_cost": result.total_cost,
                })
            sensitivity["cost_per_contact"] = pd.DataFrame(results)
        
        # Sensitivity to LTV
        if "avg_ltv" in param_ranges:
            results = []
            original_ltv = self.avg_ltv
            for ltv in param_ranges["avg_ltv"]:
                self.avg_ltv = ltv
                result = self.simulate(churn_proba, base, risk_threshold)
                results.append({
                    "avg_ltv": ltv,
                    "roi": result.roi,
                    "incremental_revenue": result.incremental_revenue,
                    "expected_saves_revenue": result.expected_saves_revenue,
                })
            self.avg_ltv = original_ltv
            sensitivity["avg_ltv"] = pd.DataFrame(results)
        
        return sensitivity
    
    def generate_targeting_list(
        self,
        customer_ids: pd.Series,
        churn_proba: pd.Series,
        segments: Optional[pd.Series] = None,
        risk_threshold: float = 0.5,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate a prioritized list of customers to target.
        
        Args:
            customer_ids: Customer ID series
            churn_proba: Churn probabilities
            segments: Optional segment labels
            risk_threshold: Minimum risk to include
            top_n: Limit to top N customers
            
        Returns:
            DataFrame with targeting list
        """
        df = pd.DataFrame({
            "customer_id": customer_ids.values,
            "churn_risk": churn_proba.values,
        })
        
        if segments is not None:
            df["segment"] = segments.values
        
        # Filter and sort
        df = df[df["churn_risk"] >= risk_threshold]
        df = df.sort_values("churn_risk", ascending=False)
        
        if top_n:
            df = df.head(top_n)
        
        # Add priority tier
        df["priority"] = pd.qcut(
            df["churn_risk"].rank(method="first"),
            q=3,
            labels=["Medium", "High", "Critical"]
        )
        
        df["expected_save_value"] = df["churn_risk"] * self.avg_ltv
        
        return df.reset_index(drop=True)


def quick_roi_estimate(
    n_customers: int,
    avg_churn_rate: float,
    avg_ltv: float,
    cost_per_contact: float = 5.0,
    expected_lift: float = 0.20,
    success_rate: float = 0.30,
) -> Dict:
    """
    Quick back-of-envelope ROI calculation.
    
    Args:
        n_customers: Number of customers to target
        avg_churn_rate: Average churn probability in target group
        avg_ltv: Average customer lifetime value
        cost_per_contact: Cost per outreach
        expected_lift: Expected churn reduction
        success_rate: Response rate
        
    Returns:
        Dictionary with ROI metrics
    """
    # Baseline churners
    expected_churners = n_customers * avg_churn_rate
    
    # Saves
    expected_saves = expected_churners * expected_lift * success_rate
    
    # Costs
    total_cost = n_customers * cost_per_contact
    
    # Revenue
    revenue_from_saves = expected_saves * avg_ltv
    
    # ROI
    incremental = revenue_from_saves - total_cost
    roi = incremental / total_cost if total_cost > 0 else 0
    
    return {
        "targeted_customers": n_customers,
        "expected_churners_baseline": int(expected_churners),
        "expected_saves": int(expected_saves),
        "campaign_cost": f"${total_cost:,.0f}",
        "revenue_from_saves": f"${revenue_from_saves:,.0f}",
        "incremental_revenue": f"${incremental:,.0f}",
        "roi": f"{roi:.0%}",
        "profitable": incremental > 0,
    }
