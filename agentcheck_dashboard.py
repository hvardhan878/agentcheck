#!/usr/bin/env python3
"""AgentCheck Analytics Dashboard - A beautiful web GUI for trace and testing analytics."""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="AgentCheck Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .warning-card {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .error-card {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def load_traces_from_directory(directory: Path) -> List[Dict[str, Any]]:
    """Load all trace files from a directory."""
    traces = []
    if not directory.exists():
        return traces
    
    for file_path in directory.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add file info
                data['_file_path'] = str(file_path)
                data['_file_name'] = file_path.name
                traces.append(data)
        except Exception as e:
            st.sidebar.warning(f"Could not load {file_path.name}: {e}")
    
    return traces


def analyze_trace_data(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trace data and return summary statistics."""
    if not traces:
        return {}
    
    total_traces = len(traces)
    total_cost = sum(trace.get('metadata', {}).get('total_cost', 0) for trace in traces)
    total_steps = sum(len(trace.get('steps', [])) for trace in traces)
    
    # Error analysis
    error_traces = [t for t in traces if t.get('metadata', {}).get('exception')]
    error_rate = len(error_traces) / total_traces if total_traces > 0 else 0
    
    # LLM call analysis
    llm_calls = []
    for trace in traces:
        for step in trace.get('steps', []):
            if step.get('type') == 'llm_call':
                llm_calls.append(step)
    
    # Model usage
    models_used = {}
    for call in llm_calls:
        model = call.get('input', {}).get('model', 'unknown')
        models_used[model] = models_used.get(model, 0) + 1
    
    # Time analysis
    durations = []
    for trace in traces:
        start = trace.get('start_time')
        end = trace.get('end_time')
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
                durations.append(duration)
            except:
                pass
    
    return {
        'total_traces': total_traces,
        'total_cost': total_cost,
        'total_steps': total_steps,
        'error_rate': error_rate,
        'error_count': len(error_traces),
        'llm_calls': len(llm_calls),
        'models_used': models_used,
        'avg_duration': np.mean(durations) if durations else 0,
        'durations': durations,
        'traces_by_date': group_traces_by_date(traces),
    }


def group_traces_by_date(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group traces by date."""
    by_date = {}
    for trace in traces:
        start_time = trace.get('start_time', '')
        if start_time:
            try:
                date = datetime.fromisoformat(start_time.replace('Z', '+00:00')).date()
                date_str = date.strftime('%Y-%m-%d')
                by_date[date_str] = by_date.get(date_str, 0) + 1
            except:
                pass
    return by_date


def load_baseline_data(directory: Path) -> List[Dict[str, Any]]:
    """Load baseline and failure report data."""
    baselines = []
    if not directory.exists():
        return baselines
    
    for file_path in directory.glob("*_baseline.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = str(file_path)
                data['_file_name'] = file_path.name
                baselines.append(data)
        except Exception as e:
            st.sidebar.warning(f"Could not load baseline {file_path.name}: {e}")
    
    return baselines


def load_failure_reports(directory: Path) -> List[Dict[str, Any]]:
    """Load failure report data."""
    reports = []
    if not directory.exists():
        return reports
    
    for file_path in directory.glob("failure_report_*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = str(file_path)
                data['_file_name'] = file_path.name
                reports.append(data)
        except Exception as e:
            st.sidebar.warning(f"Could not load failure report {file_path.name}: {e}")
    
    return reports


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AgentCheck Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Trace ⋅ Replay ⋅ Test your AI agents like real software**")
    
    # Sidebar configuration
    st.sidebar.header("📂 Data Sources")
    
    # Directory selection
    trace_dir = st.sidebar.text_input(
        "Trace Directory", 
        value="traces",
        help="Directory containing trace JSON files"
    )
    
    baseline_dir = st.sidebar.text_input(
        "Baseline Directory", 
        value="baselines",
        help="Directory containing baseline and failure report files"
    )
    
    # Load data
    traces = load_traces_from_directory(Path(trace_dir))
    baselines = load_baseline_data(Path(baseline_dir))
    failure_reports = load_failure_reports(Path(baseline_dir))
    
    # Analysis
    trace_analysis = analyze_trace_data(traces)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Trace Analysis", "🧪 Deterministic Testing", "💰 Cost Analysis"])
    
    with tab1:
        st.header("📊 System Overview")
        
        if not traces and not baselines:
            st.warning("No data found. Make sure your trace and baseline directories contain JSON files.")
            st.info("💡 **Quick Start:**\n1. Run some agents with agentcheck tracing\n2. Set up deterministic baselines\n3. Refresh this dashboard")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{trace_analysis.get('total_traces', 0)}</h3>
                <p>Total Traces</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>${trace_analysis.get('total_cost', 0):.4f}</h3>
                <p>Total Cost</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            error_rate = trace_analysis.get('error_rate', 0)
            card_class = "error-card" if error_rate > 0.1 else "warning-card" if error_rate > 0.05 else "success-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{error_rate:.1%}</h3>
                <p>Error Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(baselines)}</h3>
                <p>Baselines</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Traces Over Time")
            if trace_analysis.get('traces_by_date'):
                dates = list(trace_analysis['traces_by_date'].keys())
                counts = list(trace_analysis['traces_by_date'].values())
                
                fig = px.line(
                    x=dates, y=counts,
                    title="Daily Trace Count",
                    labels={'x': 'Date', 'y': 'Number of Traces'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time series data available")
        
        with col2:
            st.subheader("🤖 Model Usage")
            models = trace_analysis.get('models_used', {})
            if models:
                fig = px.pie(
                    values=list(models.values()),
                    names=list(models.keys()),
                    title="LLM Model Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No model usage data available")
    
    with tab2:
        st.header("🔍 Detailed Trace Analysis")
        
        if not traces:
            st.warning("No traces found in the specified directory.")
            return
        
        # Trace selection
        trace_names = [f"{t['_file_name']} ({t.get('metadata', {}).get('function_name', 'unknown')})" for t in traces]
        selected_trace_idx = st.selectbox("Select Trace", range(len(traces)), format_func=lambda x: trace_names[x])
        
        if selected_trace_idx is not None:
            trace = traces[selected_trace_idx]
            
            # Trace details
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Trace Metadata")
                metadata = trace.get('metadata', {})
                st.json({
                    'Trace ID': trace.get('trace_id', 'N/A'),
                    'Function': metadata.get('function_name', 'N/A'),
                    'Start Time': trace.get('start_time', 'N/A'),
                    'End Time': trace.get('end_time', 'N/A'),
                    'Total Cost': f"${metadata.get('total_cost', 0):.4f}",
                    'Steps': len(trace.get('steps', [])),
                })
            
            with col2:
                st.subheader("⚡ Performance Metrics")
                steps = trace.get('steps', [])
                llm_steps = [s for s in steps if s.get('type') == 'llm_call']
                
                perf_data = {
                    'Total Steps': len(steps),
                    'LLM Calls': len(llm_steps),
                    'Avg Response Time': f"{trace_analysis.get('avg_duration', 0):.2f}s",
                    'Has Errors': '❌ Yes' if trace.get('metadata', {}).get('exception') else '✅ No',
                }
                st.json(perf_data)
            
            # Step breakdown
            st.subheader("🔄 Step-by-Step Execution")
            if steps:
                step_data = []
                for i, step in enumerate(steps):
                    step_data.append({
                        'Step': i + 1,
                        'Type': step.get('type', 'unknown'),
                        'Duration': 'N/A',  # Could calculate from timestamps
                        'Status': '❌ Error' if step.get('error') else '✅ Success',
                        'Cost': f"${step.get('output', {}).get('cost', 0):.4f}",
                    })
                
                df = pd.DataFrame(step_data)
                st.dataframe(df, use_container_width=True)
                
                # Show detailed step info
                selected_step = st.selectbox("View Step Details", range(len(steps)), format_func=lambda x: f"Step {x+1}: {steps[x].get('type', 'unknown')}")
                if selected_step is not None:
                    st.json(steps[selected_step])
            else:
                st.info("No steps recorded in this trace")
    
    with tab3:
        st.header("🧪 Deterministic Testing Results")
        
        # Baseline overview
        if baselines:
            st.subheader("📊 Baseline Summary")
            baseline_data = []
            for baseline in baselines:
                baseline_data.append({
                    'Name': baseline.get('baseline_name', 'Unknown'),
                    'Created': baseline.get('created_at', 'N/A'),
                    'Test Inputs': len(baseline.get('test_inputs', [])),
                    'Baseline Runs': baseline.get('baseline_runs', 0),
                    'Threshold': f"{baseline.get('consistency_threshold', 0):.1%}",
                })
            
            df = pd.DataFrame(baseline_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No baselines found. Run deterministic testing to see results here.")
        
        # Failure reports
        if failure_reports:
            st.subheader("❌ Recent Test Failures")
            for report in failure_reports[-5:]:  # Show last 5 reports
                with st.expander(f"Failure Report - {report.get('baseline_name', 'Unknown')} ({report.get('test_timestamp', 'N/A')})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Failed Tests", f"{report.get('failed_tests', 0)}/{report.get('total_tests', 0)}")
                        st.metric("Threshold", f"{report.get('threshold', 0):.1%}")
                    
                    with col2:
                        success_rate = ((report.get('total_tests', 0) - report.get('failed_tests', 0)) / report.get('total_tests', 1)) * 100
                        st.metric("Success Rate", f"{success_rate:.1f}%")
                    
                    # Show individual failures
                    failures = report.get('failures', [])
                    if failures:
                        st.subheader("Individual Failures")
                        failure_data = []
                        for failure in failures:
                            failure_data.append({
                                'Input': str(failure.get('input_data', ''))[:50] + '...',
                                'Consistency Score': f"{failure.get('consistency_score', 0):.3f}",
                                'Threshold': f"{failure.get('threshold', 0):.3f}",
                                'Status': '❌ Failed',
                            })
                        
                        df = pd.DataFrame(failure_data)
                        st.dataframe(df, use_container_width=True)
        else:
            st.info("No failure reports found.")
        
        # Consistency trends
        if failure_reports:
            st.subheader("📈 Consistency Trends")
            trend_data = []
            for report in failure_reports:
                timestamp = report.get('test_timestamp', '')
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        success_rate = ((report.get('total_tests', 0) - report.get('failed_tests', 0)) / report.get('total_tests', 1)) * 100
                        trend_data.append({
                            'Date': date,
                            'Success Rate': success_rate,
                            'Baseline': report.get('baseline_name', 'Unknown'),
                        })
                    except:
                        pass
            
            if trend_data:
                df = pd.DataFrame(trend_data)
                fig = px.line(
                    df, x='Date', y='Success Rate', color='Baseline',
                    title="Consistency Success Rate Over Time",
                    labels={'Success Rate': 'Success Rate (%)'}
                )
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("💰 Cost Analysis")
        
        if not traces:
            st.info("No cost data available.")
            return
        
        # Cost overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_cost = trace_analysis.get('total_cost', 0)
            st.metric("Total Cost", f"${total_cost:.4f}")
        
        with col2:
            avg_cost = total_cost / trace_analysis.get('total_traces', 1)
            st.metric("Average Cost per Trace", f"${avg_cost:.4f}")
        
        with col3:
            llm_calls = trace_analysis.get('llm_calls', 0)
            avg_cost_per_call = total_cost / llm_calls if llm_calls > 0 else 0
            st.metric("Average Cost per LLM Call", f"${avg_cost_per_call:.4f}")
        
        # Cost breakdown by model
        st.subheader("💸 Cost by Model")
        model_costs = {}
        for trace in traces:
            for step in trace.get('steps', []):
                if step.get('type') == 'llm_call':
                    model = step.get('input', {}).get('model', 'unknown')
                    cost = step.get('output', {}).get('cost', 0)
                    model_costs[model] = model_costs.get(model, 0) + cost
        
        if model_costs:
            fig = px.bar(
                x=list(model_costs.keys()),
                y=list(model_costs.values()),
                title="Total Cost by Model",
                labels={'x': 'Model', 'y': 'Cost ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cost over time
        st.subheader("📈 Cost Trends")
        daily_costs = {}
        for trace in traces:
            start_time = trace.get('start_time', '')
            cost = trace.get('metadata', {}).get('total_cost', 0)
            if start_time and cost > 0:
                try:
                    date = datetime.fromisoformat(start_time.replace('Z', '+00:00')).date()
                    date_str = date.strftime('%Y-%m-%d')
                    daily_costs[date_str] = daily_costs.get(date_str, 0) + cost
                except:
                    pass
        
        if daily_costs:
            dates = list(daily_costs.keys())
            costs = list(daily_costs.values())
            
            fig = px.line(
                x=dates, y=costs,
                title="Daily Costs",
                labels={'x': 'Date', 'y': 'Cost ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cost trend data available")
    
    # Footer
    st.markdown("---")
    st.markdown("🤖 **AgentCheck Analytics Dashboard** | Built with ❤️ using Streamlit")


if __name__ == "__main__":
    main() 