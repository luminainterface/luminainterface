#!/bin/bash

# Create milestone for Phase 1 issues
gh api repos/:owner/:repo/milestones -f title="Q2-Blocking" -f description="High-priority gaps that must be addressed in Q2" -f due_on="2024-06-30T23:59:59Z"

# Create issues
gh issue create \
  --title "Implement Data Retention Policy" \
  --body "## Overview
Implement comprehensive data retention policy for all data types in the system.

## Requirements
- Define retention periods for all data types
- Implement purge scripts
- Set up backup verification
- Document compliance requirements

## Acceptance Criteria
- [ ] Retention matrix approved
- [ ] S3 lifecycle rules applied
- [ ] Collection restore time < 5min
- [ ] Documentation updated

## References
- [Data Retention Policy](/docs/data-retention.yml)
- [Gap Status](/docs/GAP_STATUS.md)" \
  --label "data-governance" \
  --assignee "@backend-lead" \
  --milestone "Q2-Blocking"

gh issue create \
  --title "Implement Backup Verification Procedures" \
  --body "## Overview
Set up automated backup verification and monitoring for critical data stores.

## Requirements
- Implement backup verification cron job
- Set up monitoring and alerting
- Create restore testing procedure
- Document backup/restore process

## Acceptance Criteria
- [ ] Daily backup verification
- [ ] Automated restore testing
- [ ] Backup success metrics
- [ ] Alert on backup failure

## References
- [Gap Status](/docs/GAP_STATUS.md)" \
  --label "ops" \
  --assignee "@devops-lead" \
  --milestone "Q2-Blocking"

gh issue create \
  --title "Create Incident Response Playbooks" \
  --body "## Overview
Develop comprehensive incident response playbooks for all critical services.

## Requirements
- Create runbook template
- Document common failure modes
- Define escalation paths
- Set up communication templates

## Acceptance Criteria
- [ ] Runbook template created
- [ ] Common issues documented
- [ ] Escalation matrix defined
- [ ] Communication templates ready

## References
- [Runbook Template](/docs/runbooks/_template.md)
- [Gap Status](/docs/GAP_STATUS.md)" \
  --label "ops" \
  --assignee "@sre-lead" \
  --milestone "Q2-Blocking"

gh issue create \
  --title "Implement Centralized Logging Solution" \
  --body "## Overview
Set up centralized logging infrastructure using Loki/Promtail.

## Requirements
- Deploy Loki stack
- Configure log shipping
- Set up log retention
- Create log analysis tools

## Acceptance Criteria
- [ ] Loki stack deployed
- [ ] All services shipping logs
- [ ] Log retention configured
- [ ] Basic dashboards created

## References
- [Gap Status](/docs/GAP_STATUS.md)" \
  --label "observability" \
  --assignee "@devops-lead" \
  --milestone "Q2-Blocking"

gh issue create \
  --title "Implement Cost Control Measures" \
  --body "## Overview
Set up cost monitoring and control measures for all services.

## Requirements
- Implement budget alerts
- Set up cost tracking
- Create cost optimization plan
- Document cost allocation

## Acceptance Criteria
- [ ] Budget alerts at 80% spend
- [ ] Cost per 1k nodes tracked
- [ ] Cost optimization plan
- [ ] Cost allocation documented

## References
- [Gap Status](/docs/GAP_STATUS.md)" \
  --label "finops" \
  --assignee "@eng-manager" \
  --milestone "Q2-Blocking"

echo "Created 5 high-priority issues for Phase 1" 