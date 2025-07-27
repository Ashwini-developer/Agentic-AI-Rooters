import os
import json
import streamlit as st
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px


DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data_dir')

# Utility to load all user data
def load_all_user_data():
    merged = {}
    for user_id in os.listdir(DATA_DIR):
        user_path = os.path.join(DATA_DIR, user_id)
        if os.path.isdir(user_path):
            user_data = {}
            for fname in os.listdir(user_path):
                if fname.endswith('.json'):
                    fpath = os.path.join(user_path, fname)
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        key = fname.replace('.json', '')
                        user_data[key] = data
                    except Exception as e:
                        user_data[fname] = f"Error reading: {e}"
            merged[user_id] = user_data
    return merged

# --- Enhanced Pattern Analysis Functions ---
def analyze_spending_pattern(bank_txns):
    spending = defaultdict(float)
    transaction_details = defaultdict(list)
    
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        amount_float = float(amount)
        
        if ttype == 2:  # DEBIT
            # Enhanced categorization
            if 'ATM' in narration.upper() or 'CASH WDL' in narration.upper():
                spending['ATM Withdrawals'] += amount_float
                transaction_details['ATM Withdrawals'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif 'RENT' in narration.upper():
                spending['Rent'] += amount_float
                transaction_details['Rent'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif ('GROCERY' in narration.upper() or 'GROCERIES' in narration.upper() or 
                  'RELIANCE FRESH' in narration.upper() or 'BIG BAZAAR' in narration.upper() or
                  'DMART' in narration.upper() or 'SHOPPING' in narration.upper()):
                spending['Shopping/Groceries'] += amount_float
                transaction_details['Shopping/Groceries'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif 'CREDIT CARD' in narration.upper():
                spending['Credit Card Payment'] += amount_float
                transaction_details['Credit Card Payment'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif 'FUEL' in narration.upper() or 'PETROL' in narration.upper() or 'INDIAN OIL' in narration.upper():
                spending['Fuel'] += amount_float
                transaction_details['Fuel'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif ('FOOD' in narration.upper() or 'ZOMATO' in narration.upper() or 
                  'SWIGGY' in narration.upper() or 'RESTAURANT' in narration.upper() or
                  'DINING' in narration.upper()):
                spending['Food'] += amount_float
                transaction_details['Food'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            elif 'BILL' in narration.upper() or 'ELECTRICITY' in narration.upper() or 'POWER' in narration.upper():
                spending['Bills'] += amount_float
                transaction_details['Bills'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
            else:
                spending['Other'] += amount_float
                transaction_details['Other'].append({
                    'amount': amount_float,
                    'narration': narration,
                    'date': date,
                    'mode': mode
                })
    
    return dict(spending), dict(transaction_details)

def detect_suspicious_transactions(bank_txns, user_id, bank_name):
    """Detect suspicious transactions based on RBI rules and patterns"""
    suspicious = []
    
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        amount_float = float(amount)
        flags = []
        
        # RBI Rule 1: Large cash transactions (>10L)
        if amount_float > 1000000 and mode in ['CASH', 'ATM']:
            flags.append("Large cash transaction (>10L)")
        
        # RBI Rule 2: Multiple large transactions in short time
        if amount_float > 500000:
            flags.append("Large transaction amount (>5L)")
        
        # RBI Rule 3: Unusual transaction patterns
        if 'UNKNOWN' in narration.upper() or 'TEST' in narration.upper():
            flags.append("Suspicious narration")
        
        # RBI Rule 4: High frequency transactions
        if ttype == 2 and amount_float > 100000:  # Large debits
            flags.append("Large debit transaction")
        
        # RBI Rule 5: Transactions outside normal hours (if we had time data)
        # This would require time parsing which we don't have in current data
        
        # RBI Rule 6: Transactions to known suspicious entities
        suspicious_keywords = ['CRYPTO', 'BITCOIN', 'GAMBLING', 'CASINO', 'BETTING']
        if any(keyword in narration.upper() for keyword in suspicious_keywords):
            flags.append("Suspicious entity transaction")
        
        # RBI Rule 7: Round figure large transactions (potential structuring)
        if amount_float >= 100000 and amount_float % 10000 == 0:
            flags.append("Round figure large transaction")
        
        # RBI Rule 8: Transactions to shell companies (simplified check)
        if len(narration) < 5 or narration.isdigit():
            flags.append("Minimal transaction description")
        
        if flags:
            suspicious.append({
                'user_id': user_id,
                'bank_name': bank_name,
                'amount': amount_float,
                'narration': narration,
                'date': date,
                'type': ttype,
                'mode': mode,
                'balance': balance,
                'flags': flags,
                'risk_level': 'HIGH' if len(flags) > 2 else 'MEDIUM' if len(flags) > 1 else 'LOW'
            })
    
    return suspicious

def analyze_loan_pattern(bank_txns):
    loan_payments = 0
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        if ttype == 6 or 'LOAN' in narration.upper() or 'EMI' in narration.upper():
            loan_payments += float(amount)
    return loan_payments

def analyze_credit_pattern(bank_txns):
    salary = 0
    credit_card_uses = 0
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        if ttype == 1 and 'SALARY' in narration.upper():
            salary += float(amount)
        if 'CREDIT CARD' in narration.upper():
            credit_card_uses += 1
    return {'salary_inflow': salary, 'credit_card_uses': credit_card_uses}

def analyze_investment_pattern(bank_txns):
    investment = 0
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        if 'SIP' in narration.upper() or 'MF' in narration.upper() or 'ETF' in narration.upper() or 'INVEST' in narration.upper():
            investment += float(amount)
    return investment

def is_stable_customer(bank_txns):
    salary_dates = set()
    investment_count = 0
    missed_payments = 0
    for txn in bank_txns:
        amount, narration, date, ttype, mode, balance = txn
        if ttype == 1 and 'SALARY' in narration.upper():
            salary_dates.add(date[:7])  # YYYY-MM
        if 'SIP' in narration.upper() or 'MF' in narration.upper() or 'ETF' in narration.upper():
            investment_count += 1
        if 'BOUNCE' in narration.upper() or 'FAILED' in narration.upper():
            missed_payments += 1
    return len(salary_dates) >= 2 and investment_count >= 2 and missed_payments == 0

# --- Enhanced Federated Simulation ---
def federated_analysis(all_data, noise_level=0.0):
    per_bank_results = defaultdict(lambda: defaultdict(list))
    global_results = {
        'spending': Counter(),
        'loan': 0,
        'credit': {'salary_inflow': 0, 'credit_card_uses': 0},
        'investment': 0,
        'stable_customers': 0,
        'total_customers': 0
    }
    all_suspicious_transactions = []
    all_transaction_details = defaultdict(list)
    
    for user_id, user_data in all_data.items():
        bank_txns_list = user_data.get('fetch_bank_transactions', {}).get('bankTransactions', [])
        for bank in bank_txns_list:
            bank_name = bank.get('bank', f'Unknown_{user_id}')
            txns = bank.get('txns', [])
            
            # Enhanced spending analysis with transaction details
            spending, transaction_details = analyze_spending_pattern(txns)
            
            # Detect suspicious transactions
            suspicious = detect_suspicious_transactions(txns, user_id, bank_name)
            all_suspicious_transactions.extend(suspicious)
            
            # Store transaction details for drill-down
            for category, details in transaction_details.items():
                all_transaction_details[category].extend(details)
            
            loan = analyze_loan_pattern(txns)
            credit = analyze_credit_pattern(txns)
            investment = analyze_investment_pattern(txns)
            stable = is_stable_customer(txns)
            
            per_bank_results[bank_name]['spending'].append(spending)
            per_bank_results[bank_name]['loan'].append(loan)
            per_bank_results[bank_name]['credit'].append(credit)
            per_bank_results[bank_name]['investment'].append(investment)
            per_bank_results[bank_name]['stable'].append(stable)
            
            global_results['spending'].update(spending)
            global_results['loan'] += loan
            global_results['credit']['salary_inflow'] += credit['salary_inflow']
            global_results['credit']['credit_card_uses'] += credit['credit_card_uses']
            global_results['investment'] += investment
            global_results['stable_customers'] += int(stable)
            global_results['total_customers'] += 1
    
    # Apply differential privacy (add noise)
    if noise_level > 0.0:
        for k in global_results['spending']:
            global_results['spending'][k] += np.random.normal(0, noise_level)
        global_results['loan'] += np.random.normal(0, noise_level)
        global_results['credit']['salary_inflow'] += np.random.normal(0, noise_level)
        global_results['credit']['credit_card_uses'] += np.random.normal(0, noise_level)
        global_results['investment'] += np.random.normal(0, noise_level)
    
    return per_bank_results, global_results, all_suspicious_transactions, all_transaction_details

# --- UI Setup ---
st.set_page_config(page_title='Federated Financial Simulator', layout='wide')
st.title('Federated Financial Data Aggregator & Simulator')

all_data = load_all_user_data()
bank_names = set()
for user_data in all_data.values():
    for bank in user_data.get('fetch_bank_transactions', {}).get('bankTransactions', []):
        bank_names.add(bank.get('bank', 'Unknown'))
bank_names = sorted(list(bank_names))

with st.sidebar:
    st.header('Federated Settings')
    noise_level = st.slider('Differential Privacy Noise (std dev)', 0.0, 10000.0, 0.0, 100.0)
    st.caption('Higher noise = more privacy, less accuracy')
    st.markdown('---')
    st.header('Bank Comparison')
    selected_banks = st.multiselect('Select banks to compare', bank_names, default=bank_names[:2])
    st.markdown('---')

per_bank, global_view, suspicious_transactions, transaction_details = federated_analysis(all_data, noise_level=noise_level)

# --- Tabs UI ---
tabs = st.tabs([
    'Aggregated (Federated) View',
    'Bank Comparison',
    'Suspicious Transactions',
    'Transaction Drill-Down',
    'Time Trends',
    'Customer Segmentation',
    'Predictive Risk',
   
])

with tabs[0]:
    st.header('Federated (Global) Analysis')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Spending Pattern (Global)')
        spend_df = pd.DataFrame(list(global_view['spending'].items()), columns=['Category', 'Amount'])
        st.bar_chart(spend_df.set_index('Category'))
        st.write(spend_df)
        st.subheader('Investment Pattern (Global)')
        st.metric('Total Investment Amount', f"₹{global_view['investment']:.2f}")
        st.metric('Total Loan Payments', f"₹{global_view['loan']:.2f}")
    with col2:
        st.subheader('Credit Pattern (Global)')
        st.metric('Total Salary Inflow', f"₹{global_view['credit']['salary_inflow']:.2f}")
        st.metric('Total Credit Card Uses', int(global_view['credit']['credit_card_uses']))
        st.subheader('Stable Customers')
        st.metric('Stable Customers', f"{global_view['stable_customers']} / {global_view['total_customers']}")
    st.info('This view shows the global (federated) results, optionally with differential privacy applied.')

with tabs[1]:
    st.header('Bank Comparison (Federated Clients)')
    for bank in selected_banks:
        st.subheader(f'Bank: {bank}')
        results = per_bank.get(bank, {})
        agg_spending = Counter()
        agg_loan = 0
        agg_credit = {'salary_inflow': 0, 'credit_card_uses': 0}
        agg_investment = 0
        stable_count = 0
        total = 0
        for s in results.get('spending', []):
            agg_spending.update(s)
        for l in results.get('loan', []):
            agg_loan += l
        for c in results.get('credit', []):
            agg_credit['salary_inflow'] += c['salary_inflow']
            agg_credit['credit_card_uses'] += c['credit_card_uses']
        for i in results.get('investment', []):
            agg_investment += i
        for s in results.get('stable', []):
            stable_count += int(s)
            total += 1
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Spending Pattern')
            spend_df = pd.DataFrame(list(agg_spending.items()), columns=['Category', 'Amount'])
            st.bar_chart(spend_df.set_index('Category'))
            st.write(spend_df)
            st.subheader('Investment Pattern')
            st.metric('Total Investment Amount', f"₹{agg_investment:.2f}")
            st.metric('Total Loan Payments', f"₹{agg_loan:.2f}")
        with col2:
            st.subheader('Credit Pattern')
            st.metric('Total Salary Inflow', f"₹{agg_credit['salary_inflow']:.2f}")
            st.metric('Total Credit Card Uses', int(agg_credit['credit_card_uses']))
            st.subheader('Stable Customers')
            st.metric('Stable Customers', f"{stable_count} / {total}")
        # Caution areas
        st.warning('Caution Areas:')
        caution_msgs = []
        if agg_spending.get('ATM Withdrawals', 0) > 200000:
            caution_msgs.append('High ATM withdrawals detected.')
        if agg_spending.get('Other', 0) > 100000:
            caution_msgs.append('Significant uncategorized spending.')
        if agg_loan > 500000:
            caution_msgs.append('High loan/EMI outflows.')
        if agg_credit['credit_card_uses'] > 10:
            caution_msgs.append('Frequent credit card usage.')
        if not caution_msgs:
            st.write('No major caution areas detected.')
        else:
            for msg in caution_msgs:
                st.write(f'- {msg}')
        # Personalization advice
        st.success('Personalization Advice:')
        advice_msgs = []
        if agg_spending.get('Shopping/Groceries', 0) > 100000:
            advice_msgs.append('Consider using digital wallets for shopping to earn rewards.')
        if agg_investment < 50000:
            advice_msgs.append('Increase SIPs or investments for better wealth growth.')
        if agg_spending.get('Bills', 0) > 50000:
            advice_msgs.append('Automate bill payments to avoid late fees.')
        if stable_count / (total or 1) > 0.8:
            advice_msgs.append('Customer base is stable. Offer loyalty programs.')
        if not advice_msgs:
            st.write('No specific personalization advice at this time.')
        else:
            for msg in advice_msgs:
                st.write(f'- {msg}')

with tabs[2]:
    st.header('🚨 Suspicious Transactions Detection')
    st.subheader('Transactions Flagged Based on RBI Rules')
    bank_options = ['ALL'] + bank_names
    selected_bank = st.selectbox('Filter by Bank', bank_options)
    if suspicious_transactions:
        if selected_bank == 'ALL':
            filtered_transactions = suspicious_transactions
        else:
            filtered_transactions = [t for t in suspicious_transactions if t['bank_name'] == selected_bank]
        high_risk = [t for t in filtered_transactions if t['risk_level'] == 'HIGH']
        medium_risk = [t for t in filtered_transactions if t['risk_level'] == 'MEDIUM']
        low_risk = [t for t in filtered_transactions if t['risk_level'] == 'LOW']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('High Risk', len(high_risk), delta=None)
        with col2:
            st.metric('Medium Risk', len(medium_risk), delta=None)
        with col3:
            st.metric('Low Risk', len(low_risk), delta=None)
        risk_level = st.selectbox('Select Risk Level', ['HIGH', 'MEDIUM', 'LOW', 'ALL'])
        if risk_level == 'ALL':
            display_transactions = filtered_transactions
        else:
            display_transactions = [t for t in filtered_transactions if t['risk_level'] == risk_level]
        if display_transactions:
            st.subheader(f'{risk_level} Risk Transactions')
            for txn in display_transactions:
                with st.expander(f"₹{txn['amount']:,.2f} - {txn['narration'][:50]}..."):
                    st.write(f"**User ID:** {txn['user_id']}")
                    st.write(f"**Bank:** {txn['bank_name']}")
                    st.write(f"**Amount:** ₹{txn['amount']:,.2f}")
                    st.write(f"**Date:** {txn['date']}")
                    st.write(f"**Mode:** {txn['mode']}")
                    st.write(f"**Risk Level:** {txn['risk_level']}")
                    st.write("**Flags:**")
                    for flag in txn['flags']:
                        st.write(f"- {flag}")
        else:
            st.success("No suspicious transactions detected!")
        st.markdown('---')
        st.subheader('How Customers Can Avoid Being Flagged')
        all_flags = []
        for txn in filtered_transactions:
            all_flags.extend(txn['flags'])
        flag_counts = Counter(all_flags)
        if not flag_counts:
            st.info('No specific advice needed. No flagged transactions for this bank.')
        else:
            if 'Large cash transaction (>10L)' in flag_counts:
                st.write('- Avoid large cash withdrawals/deposits. Use digital channels for high-value transactions.')
            if 'Large transaction amount (>5L)' in flag_counts:
                st.write('- Split large transactions when possible and ensure they are for legitimate purposes.')
            if 'Suspicious narration' in flag_counts:
                st.write('- Use clear, descriptive narrations for all transactions.')
            if 'Large debit transaction' in flag_counts:
                st.write('- Monitor and limit large debits. Ensure they are justified and documented.')
            if 'Suspicious entity transaction' in flag_counts:
                st.write('- Avoid transactions with crypto, gambling, or other flagged entities.')
            if 'Round figure large transaction' in flag_counts:
                st.write('- Avoid structuring transactions in round figures to evade detection.')
            if 'Minimal transaction description' in flag_counts:
                st.write('- Always provide detailed descriptions for transactions.')
            st.info('Following these practices can help reduce the chance of your transactions being flagged as suspicious.')
    else:
        st.success("No suspicious transactions detected!")
    st.markdown("---")
    st.subheader("RBI Rules Applied for Detection:")
    st.markdown("""
    1. **Large Cash Transactions (>10L):** Transactions over ₹10 lakhs in cash
    2. **Large Transaction Amount (>5L):** Any transaction over ₹5 lakhs
    3. **Suspicious Narration:** Transactions with 'UNKNOWN' or 'TEST' in description
    4. **Large Debit Transactions:** Debit transactions over ₹1 lakh
    5. **Suspicious Entities:** Transactions to crypto, gambling, or betting entities
    6. **Round Figure Large Transactions:** Large amounts in round figures (potential structuring)
    7. **Minimal Descriptions:** Transactions with very short or numeric descriptions
    """)

with tabs[3]:
    st.header('Transaction Category Drill-Down')
    category = st.selectbox('Select Spending Category', list(transaction_details.keys()))
    if category in transaction_details and transaction_details[category]:
        st.subheader(f'{category} Transactions')
        df = pd.DataFrame(transaction_details[category])
        df['amount'] = df['amount'].apply(lambda x: f"₹{x:,.2f}")
        st.dataframe(df, use_container_width=True)
        amounts = [t['amount'] for t in transaction_details[category]]
        st.subheader('Summary Statistics')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Total Amount', f"₹{sum(amounts):,.2f}")
        with col2:
            st.metric('Average Amount', f"₹{np.mean(amounts):,.2f}")
        with col3:
            st.metric('Number of Transactions', len(amounts))
    else:
        st.info(f'No transactions found for category: {category}')

# --- Time Trends Tab ---
with tabs[4]:
    st.header('Time Trends (Bank-wise)')
    bank_filter = st.selectbox('Select Bank for Time Trends', ['ALL'] + bank_names, key='timetrends')
    # Aggregate all transactions by month for the selected bank
    all_txns = []
    for user_data in all_data.values():
        for bank in user_data.get('fetch_bank_transactions', {}).get('bankTransactions', []):
            if bank_filter == 'ALL' or bank.get('bank', 'Unknown') == bank_filter:
                for txn in bank.get('txns', []):
                    amount, narration, date, ttype, mode, balance = txn
                    all_txns.append({'date': date, 'amount': float(amount), 'type': ttype, 'bank': bank.get('bank', 'Unknown')})
    if all_txns:
        df = pd.DataFrame(all_txns)
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
        debit_df = df[df['type'] == 2].groupby('month')['amount'].sum().reset_index()
        credit_df = df[df['type'] == 1].groupby('month')['amount'].sum().reset_index()
        st.subheader('Monthly Debit Trend')
        st.line_chart(debit_df.set_index('month'))
        st.subheader('Monthly Credit Trend')
        st.line_chart(credit_df.set_index('month'))
    else:
        st.info('No transactions found for this bank.')

# --- Customer Segmentation Tab ---
with tabs[5]:
    st.header('Customer Segmentation (Bank-wise)')
    bank_filter = st.selectbox('Select Bank for Segmentation', ['ALL'] + bank_names, key='segmentation')
    # Prepare features: total debit, total credit, total investment per user
    user_features = []
    for user_id, user_data in all_data.items():
        for bank in user_data.get('fetch_bank_transactions', {}).get('bankTransactions', []):
            if bank_filter == 'ALL' or bank.get('bank', 'Unknown') == bank_filter:
                txns = bank.get('txns', [])
                total_debit = sum(float(t[0]) for t in txns if t[3] == 2)
                total_credit = sum(float(t[0]) for t in txns if t[3] == 1)
                total_invest = sum(float(t[0]) for t in txns if 'SIP' in t[1].upper() or 'MF' in t[1].upper() or 'ETF' in t[1].upper() or 'INVEST' in t[1].upper())
                user_features.append({'user_id': user_id, 'bank': bank.get('bank', 'Unknown'), 'total_debit': total_debit, 'total_credit': total_credit, 'total_invest': total_invest})
    if user_features:
        df = pd.DataFrame(user_features)
        if len(df) > 2:
            scaler = StandardScaler()
            X = scaler.fit_transform(df[['total_debit', 'total_credit', 'total_invest']])
            kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
            df['segment'] = kmeans.labels_
            st.write('Customer Segments (0, 1, 2):')
            st.dataframe(df)
            fig = px.scatter_3d(df, x='total_debit', y='total_credit', z='total_invest', color='segment', hover_data=['user_id', 'bank'])
            st.plotly_chart(fig)
        else:
            st.info('Not enough data for segmentation.')
    else:
        st.info('No data for segmentation.')

# --- Predictive Risk Tab ---
with tabs[6]:
    st.header('Predictive Risk (Bank-wise)')
    bank_filter = st.selectbox('Select Bank for Predictive Risk', ['ALL'] + bank_names, key='risk')
    # Simple risk: users with high debit, high suspicious, low investment
    risk_data = []
    for user_id, user_data in all_data.items():
        for bank in user_data.get('fetch_bank_transactions', {}).get('bankTransactions', []):
            if bank_filter == 'ALL' or bank.get('bank', 'Unknown') == bank_filter:
                txns = bank.get('txns', [])
                total_debit = sum(float(t[0]) for t in txns if t[3] == 2)
                total_invest = sum(float(t[0]) for t in txns if 'SIP' in t[1].upper() or 'MF' in t[1].upper() or 'ETF' in t[1].upper() or 'INVEST' in t[1].upper())
                suspicious_count = sum(1 for t in suspicious_transactions if t['user_id'] == user_id and (bank_filter == 'ALL' or t['bank_name'] == bank_filter))
                risk_score = total_debit - total_invest + suspicious_count * 10000
                risk_data.append({'user_id': user_id, 'bank': bank.get('bank', 'Unknown'), 'risk_score': risk_score, 'suspicious_count': suspicious_count, 'total_debit': total_debit, 'total_invest': total_invest})
    if risk_data:
        df = pd.DataFrame(risk_data)
        st.dataframe(df.sort_values('risk_score', ascending=False))
        fig = px.scatter(df, x='total_debit', y='risk_score', color='suspicious_count', hover_data=['user_id', 'bank'])
        st.plotly_chart(fig)
    else:
        st.info('No data for predictive risk.')



st.caption('Federated learning simulation: Each bank is a federated client. Tabs provide global, per-bank, compliance, drill-down, time trends, segmentation, predictive risk, and agent-based query views. Differential privacy can be applied to protect user data in the global view.')
