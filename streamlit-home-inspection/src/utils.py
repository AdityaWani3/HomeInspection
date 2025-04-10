def parse_inspection_table(response_json: dict) -> List[Dict]:
    inspection_data = []
    for inspection in response_json['detailedInspection']:
        details = []
        media_ref = inspection['mediaReference']
        if media_ref == 'home_inspection.mp4' and inspection.get('timestamp'):
            media_ref = f"home_inspection.mp4_{inspection['timestamp']}"
        if inspection.get('condition'):
            has_issues = (inspection.get('referenceDoc') or 
                          inspection.get('referenceSection') or 
                          inspection.get('issuesFound') or 
                          inspection.get('recommendation'))
            condition_text = inspection['condition']
            if not has_issues:
                details.append(f"<strong>Condition:</strong> {condition_text}")
        if inspection.get('referenceDoc') or inspection.get('referenceSection'):
            code_reference = f"{inspection.get('referenceDoc', 'N/A')} - {inspection.get('referenceSection', 'N/A')}"
            details.append(f"<strong>Code Reference:</strong> {code_reference}") 
        if inspection.get('issuesFound'):
            issues = '<br>'.join([f"• {issue}" for issue in inspection['issuesFound']])
            details.append(f"<strong>Issues Found:</strong><br>{issues}")    
        if inspection.get('recommendation'):
            details.append(f"<strong>Recommendation:</strong> {inspection['recommendation']}")    
        details_html = '<br><br>'.join(details)
        inspection_data.append({
            'Area': inspection['area'],
            'Media': media_ref, 
            'Details': details_html,
            'Priority': 'High' if inspection['complianceStatus'] == 'Non-compliant' else 'Medium' if 'Potentially' in inspection['complianceStatus'] else 'Low'
        })
    return inspection_data

def parse_maintenance_schedule(response_json: dict) -> List[Dict]:
    schedule_items = []
    for inspection in response_json['detailedInspection']:
        if inspection['complianceStatus'] == 'Non-compliant':
            recommendation = inspection.get('recommendation', '')
            issues = inspection.get('issuesFound', [])
            frequency = 'Immediate' if any(word in ' '.join(issues).lower() 
                                             for word in ['immediate', 'critical', 'urgent', 'termite', 'pest']) \
                       else 'Quarterly'
            schedule_items.append({
                'Task': recommendation,
                'Frequency': frequency,
                'Priority': 'High' if frequency == 'Immediate' else 'Medium',
                'Status': 'Pending'
            })
    standard_tasks = [
        {
            'Task': 'General inspection of building condition',
            'Frequency': 'Annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Check and clean gutters and drainage systems',
            'Frequency': 'Quarterly',
            'Priority': 'Medium',
            'Status': 'Pending'
        },
        {
            'Task': 'Inspect for pest activity',
            'Frequency': 'Semi-annually',
            'Priority': 'Medium',
            'Status': 'Pending'
        }
    ]
    schedule_items.extend(standard_tasks)
    return schedule_items

def count_critical_issues(response_json: dict) -> Dict:
    non_compliant_count = sum(1 for item in response_json['detailedInspection'] 
                               if item['complianceStatus'] == 'Non-compliant')
    critical_issues_count = len(response_json['executiveSummary']['criticalIssues'])
    return {
        'critical_issues': non_compliant_count,
        'high_priority': critical_issues_count
    }