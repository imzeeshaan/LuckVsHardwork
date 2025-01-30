import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

def simulate_hardwork_luck(
    N=10000,
    T=15,
    alpha=0.1,
    beta=0.05,
    mu_H=0.5,
    sigma_H=0.1,
    sigma_L=0.05,
    seed=None
):
    """
    Simulates the growth of two groups of individuals over a 15-year period.
    Group A: Success determined by Hard Work only.
    Group B: Success determined by Hard Work + Luck.
    Both groups start with identical hard work distributions.
    
    Parameters:
    -----------
    N : int
        Total number of individuals to simulate (split into two groups).
    T : int
        Number of time steps (years).
    alpha : float
        Coefficient that scales the effect of hard work.
    beta : float
        Coefficient that scales the effect of luck.
    mu_H : float
        Mean value for the hard-work distribution.
    sigma_H : float
        Standard deviation for the hard-work distribution.
    sigma_L : float
        Standard deviation for the luck distribution.
    seed : int or None
        If provided, sets the random seed for reproducibility.
    
    Returns:
    --------
    S_groupA, S_groupB : np.ndarray
        Final success arrays for Group A and Group B after T years.
    """
    # Set random seed (optional) for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Split the population into two groups of equal size
    N_half = N // 2
    
    # Initialize Hard Work levels - same distribution for both groups
    H_values = np.random.normal(mu_H, sigma_H, N_half)
    H_groupA = H_values.copy()
    H_groupB = H_values.copy()
    
    # Initialize Success arrays (start everyone at 1.0)
    S_groupA = np.ones(N_half)
    S_groupB = np.ones(N_half)
    
    # Simulate year-by-year growth
    for _ in range(T):
        S_groupA *= (1 + alpha * H_groupA)
        luck_values = np.random.normal(0, sigma_L, N_half)
        S_groupB *= (1 + alpha * H_groupB + beta * luck_values)
    
    return S_groupA, S_groupB

def calculate_wealth_metrics(S_groupA, S_groupB):
    """Calculate wealth comparison metrics between groups"""
    
    # Calculate wealth ratios
    top1_A = np.percentile(S_groupA, 99)
    top1_B = np.percentile(S_groupB, 99)
    bottom50_A = np.mean(S_groupA[S_groupA <= np.median(S_groupA)])
    bottom50_B = np.mean(S_groupB[S_groupB <= np.median(S_groupB)])
    
    # Calculate wealth shares
    total_wealth_A = np.sum(S_groupA)
    total_wealth_B = np.sum(S_groupB)
    
    top10_share_A = np.sum(S_groupA[S_groupA >= np.percentile(S_groupA, 90)]) / total_wealth_A
    top10_share_B = np.sum(S_groupB[S_groupB >= np.percentile(S_groupB, 90)]) / total_wealth_B
    
    bottom50_share_A = np.sum(S_groupA[S_groupA <= np.median(S_groupA)]) / total_wealth_A
    bottom50_share_B = np.sum(S_groupB[S_groupB <= np.median(S_groupB)]) / total_wealth_B
    
    return {
        'top1_bottom50_ratio_A': top1_A / bottom50_A,
        'top1_bottom50_ratio_B': top1_B / bottom50_B,
        'top10_share_A': top10_share_A,
        'top10_share_B': top10_share_B,
        'bottom50_share_A': bottom50_share_A,
        'bottom50_share_B': bottom50_share_B
    }

def simulate_hardwork_luck_with_history(
    N=10000, T=15, alpha=0.1, beta=0.05,
    mu_H=0.5, sigma_H=0.1, sigma_L=0.05, seed=None
):
    """
    Enhanced simulation that tracks wealth growth over time
    Returns both final values and history of growth
    """
    if seed is not None:
        np.random.seed(seed)
    
    N_half = N // 2
    H_values = np.random.normal(mu_H, sigma_H, N_half)
    H_groupA = H_values.copy()
    H_groupB = H_values.copy()
    
    # Initialize arrays for tracking history
    history = {
        'time': list(range(T+1)),
        'mean_A': [1.0],
        'mean_B': [1.0],
        'median_A': [1.0],
        'median_B': [1.0],
        'top10_A': [1.0],
        'top10_B': [1.0],
        'bottom10_A': [1.0],
        'bottom10_B': [1.0]
    }
    
    S_groupA = np.ones(N_half)
    S_groupB = np.ones(N_half)
    
    # Simulate with history tracking
    for t in range(T):
        S_groupA *= (1 + alpha * H_groupA)
        luck_values = np.random.normal(0, sigma_L, N_half)
        S_groupB *= (1 + alpha * H_groupB + beta * luck_values)
        
        # Record history
        history['mean_A'].append(np.mean(S_groupA))
        history['mean_B'].append(np.mean(S_groupB))
        history['median_A'].append(np.median(S_groupA))
        history['median_B'].append(np.median(S_groupB))
        history['top10_A'].append(np.percentile(S_groupA, 90))
        history['top10_B'].append(np.percentile(S_groupB, 90))
        history['bottom10_A'].append(np.percentile(S_groupA, 10))
        history['bottom10_B'].append(np.percentile(S_groupB, 10))
    
    return S_groupA, S_groupB, history

def plot_growth_analysis(history, T):
    """Create growth analysis plots"""
    fig = plt.figure(figsize=(15, 10))
    
    # Wealth Growth Over Time
    ax1 = plt.subplot(211)
    ax1.plot(history['time'], history['mean_A'], label='Hard Work Only (Mean)', color='#2ecc71')
    ax1.plot(history['time'], history['mean_B'], label='Hard Work + Luck (Mean)', color='#3498db')
    ax1.fill_between(history['time'], history['bottom10_A'], history['top10_A'], 
                     color='#2ecc71', alpha=0.2, label='Hard Work 10-90 percentile')
    ax1.fill_between(history['time'], history['bottom10_B'], history['top10_B'], 
                     color='#3498db', alpha=0.2, label='Hard Work + Luck 10-90 percentile')
    ax1.set_xlabel('Years')
    ax1.set_ylabel('Wealth Level')
    ax1.set_title('Wealth Growth Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Inequality Growth
    ax2 = plt.subplot(212)
    inequality_A = [t/b for t, b in zip(history['top10_A'], history['bottom10_A'])]
    inequality_B = [t/b for t, b in zip(history['top10_B'], history['bottom10_B'])]
    ax2.plot(history['time'], inequality_A, label='Hard Work Only', color='#2ecc71')
    ax2.plot(history['time'], inequality_B, label='Hard Work + Luck', color='#3498db')
    ax2.set_xlabel('Years')
    ax2.set_ylabel('Top 10% / Bottom 10% Ratio')
    ax2.set_title('Inequality Growth Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    st.title("Success Simulation: Hard Work vs. Luck")
    
    # Keep basic explanation
    st.markdown("""
    This simulation compares how success evolves over time under two scenarios:
    - **Group A**: Success depends only on hard work
    - **Group B**: Success depends on both hard work and luck
    
    The simulation uses compound growth where each year's success builds on the previous year.
    """)
    
    # Keep parameter selection
    st.sidebar.header("Simulation Parameters")
    
    # Keep scenario selector
    scenario = st.sidebar.selectbox(
        "Choose a Scenario",
        ["Custom",
         "Pure Merit-Based (High Hard Work Impact)",
         "Luck-Dominated System",
         "High Volatility",
         "Long-Term Effects",
         "Balanced System (Default)"],
        help="Select a predefined scenario or customize your own"
    )
    
    # Keep scenario parameters
    if scenario == "Pure Merit-Based (High Hard Work Impact)":
        default_N = 10000
        default_T = 30
        default_alpha = 0.4
        default_beta = 0.1
        default_mu_H = 0.5
        default_sigma_H = 0.1
        default_sigma_L = 0.1
    elif scenario == "Luck-Dominated System":
        default_N = 10000
        default_T = 30
        default_alpha = 0.1
        default_beta = 0.4
        default_mu_H = 0.5
        default_sigma_H = 0.1
        default_sigma_L = 0.2
    elif scenario == "High Volatility":
        default_N = 10000
        default_T = 30
        default_alpha = 0.2
        default_beta = 0.2
        default_mu_H = 0.5
        default_sigma_H = 0.2
        default_sigma_L = 0.4
    elif scenario == "Long-Term Effects":
        default_N = 10000
        default_T = 50
        default_alpha = 0.2
        default_beta = 0.2
        default_mu_H = 0.5
        default_sigma_H = 0.1
        default_sigma_L = 0.1
    else:  # Custom or Balanced System
        default_N = 10000
        default_T = 30
        default_alpha = 0.1
        default_beta = 0.1
        default_mu_H = 0.5
        default_sigma_H = 0.1
        default_sigma_L = 0.1
    
    # Keep parameter inputs
    N = st.sidebar.slider("Population Size", 1000, 20000, default_N, 1000,
                         help="Larger populations give more stable statistical results")
    T = st.sidebar.slider("Number of Years", 5, 50, default_T,
                         help="Longer periods show compound effects more clearly")
    alpha = st.sidebar.slider("Hard Work Coefficient (α)", 0.0, 0.5, default_alpha, 0.01,
                            help="Higher values mean hard work has stronger influence")
    beta = st.sidebar.slider("Luck Coefficient (β)", 0.0, 0.5, default_beta, 0.01,
                           help="Higher values mean luck has stronger influence")
    mu_H = st.sidebar.slider("Mean Hard Work (μ)", 0.0, 1.0, default_mu_H, 0.1,
                           help="Average level of hard work in the population")
    sigma_H = st.sidebar.slider("Hard Work Std Dev (σ_H)", 0.01, 0.5, default_sigma_H, 0.01,
                              help="Variation in hard work levels")
    sigma_L = st.sidebar.slider("Luck Std Dev (σ_L)", 0.01, 0.5, default_sigma_L, 0.01,
                              help="Variation in luck events")
    
    seed = st.sidebar.number_input("Random Seed", value=42,
                                 help="Same seed produces identical results")
    
    # Run enhanced simulation
    S_groupA, S_groupB, history = simulate_hardwork_luck_with_history(
        N=N, T=T, alpha=alpha, beta=beta, mu_H=mu_H, 
        sigma_H=sigma_H, sigma_L=sigma_L, seed=seed
    )
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hard Work Only (Group A)")
        st.write(f"Mean: {np.mean(S_groupA):.2f}")
        st.write(f"Median: {np.median(S_groupA):.2f}")
    
    with col2:
        st.subheader("Hard Work + Luck (Group B)")
        st.write(f"Mean: {np.mean(S_groupB):.2f}")
        st.write(f"Median: {np.median(S_groupB):.2f}")
    
    # Basic distribution plot
    st.subheader("📊 Distribution of Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(S_groupA, bins=50, alpha=0.6, label='Hard Work Only', color='#2ecc71')
    ax.hist(S_groupB, bins=50, alpha=0.6, label='Hard Work + Luck', color='#3498db')
    ax.legend()
    ax.set_xlabel("Final Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution After {T} Years")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Add growth analysis
    st.subheader("📈 Wealth Growth Analysis")
    growth_fig = plot_growth_analysis(history, T)
    st.pyplot(growth_fig)
    
    # Add parameter impact analysis
    st.subheader("🎯 Parameter Impact Analysis")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Growth Factors")
        st.write("**Hard Work Impact**")
        st.write(f"- Contribution to growth: {alpha * mu_H:.2%} per year (average)")
        st.write(f"- Variation in contribution: ±{alpha * sigma_H:.2%}")
        
        if scenario != "Custom":
            st.markdown("### Scenario Analysis")
            if scenario == "Pure Merit-Based (High Hard Work Impact)":
                st.write("In this scenario, hard work has 4x more impact than luck")
            elif scenario == "Luck-Dominated System":
                st.write("Luck has 4x more impact than hard work")
            elif scenario == "High Volatility":
                st.write("High variation in outcomes due to increased luck volatility")
    
    with col4:
        st.markdown("### Luck Impact")
        st.write("**Random Events**")
        st.write(f"- Maximum luck impact: ±{beta * 2*sigma_L:.2%} per year (95% range)")
        st.write(f"- Typical luck variation: ±{beta * sigma_L:.2%} per year")
        
        if scenario != "Custom":
            st.markdown("### Key Metrics")
            final_inequality_A = history['top10_A'][-1] / history['bottom10_A'][-1]
            final_inequality_B = history['top10_B'][-1] / history['bottom10_B'][-1]
            st.write(f"Final inequality ratio (Group A): {final_inequality_A:.1f}x")
            st.write(f"Final inequality ratio (Group B): {final_inequality_B:.1f}x")

if __name__ == "__main__":
    main()
