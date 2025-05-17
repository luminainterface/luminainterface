const fs = require('fs');
const path = require('path');

// Load current results
const currentResults = JSON.parse(
  fs.readFileSync(path.join(process.cwd(), 'benchmark-results.json'), 'utf8')
);

// Load historical baseline if it exists
let baseline;
try {
  baseline = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), '.benchmark-baseline.json'), 'utf8')
  );
} catch (error) {
  console.log('No baseline found, using current results as baseline');
  baseline = currentResults;
  fs.writeFileSync(
    path.join(process.cwd(), '.benchmark-baseline.json'),
    JSON.stringify(currentResults, null, 2)
  );
}

// Performance regression thresholds (% increase allowed)
const REGRESSION_THRESHOLDS = {
  avgLatency: 10,
  peakMemory: 15,
  conceptThroughput: -10, // negative means decrease
  renderFps: -10,
  embedLatency: 10,
  searchLatency: 10,
  inferenceLatency: 10
};

// Absolute thresholds (from run-benchmarks.js)
const ABSOLUTE_THRESHOLDS = {
  avgLatency: 200,
  peakMemory: 384,
  conceptThroughput: 50,
  renderFps: 45,
  embedLatency: 100,
  searchLatency: 150,
  inferenceLatency: 250
};

function checkRegressions() {
  const regressions = [];
  
  // Check each metric against baseline and absolute thresholds
  for (const [metric, value] of Object.entries(currentResults.summary)) {
    const baselineValue = baseline.summary[metric];
    const percentChange = ((value - baselineValue) / baselineValue) * 100;
    
    // Check for regression against baseline
    if (Math.abs(percentChange) > Math.abs(REGRESSION_THRESHOLDS[metric])) {
      if (
        (percentChange > 0 && REGRESSION_THRESHOLDS[metric] > 0) ||
        (percentChange < 0 && REGRESSION_THRESHOLDS[metric] < 0)
      ) {
        regressions.push({
          metric,
          type: 'regression',
          current: value,
          baseline: baselineValue,
          change: percentChange.toFixed(1) + '%',
          threshold: REGRESSION_THRESHOLDS[metric] + '%'
        });
      }
    }
    
    // Check absolute threshold
    if (
      (value > ABSOLUTE_THRESHOLDS[metric] && metric !== 'conceptThroughput' && metric !== 'renderFps') ||
      (value < ABSOLUTE_THRESHOLDS[metric] && (metric === 'conceptThroughput' || metric === 'renderFps'))
    ) {
      regressions.push({
        metric,
        type: 'threshold',
        current: value,
        threshold: ABSOLUTE_THRESHOLDS[metric],
        unit: getMetricUnit(metric)
      });
    }
  }
  
  return regressions;
}

function getMetricUnit(metric) {
  if (metric.includes('Latency')) return 'ms';
  if (metric === 'peakMemory') return 'MB';
  if (metric === 'conceptThroughput') return 'concepts/s';
  if (metric === 'renderFps') return 'FPS';
  return '';
}

function generateReport(regressions) {
  if (regressions.length === 0) {
    return '✅ No performance regressions detected';
  }
  
  let report = '⚠️ Performance issues detected:\n\n';
  
  // Group by type
  const byType = regressions.reduce((acc, reg) => {
    acc[reg.type] = acc[reg.type] || [];
    acc[reg.type].push(reg);
    return acc;
  }, {});
  
  // Report regressions
  if (byType.regression) {
    report += '## Regressions from Baseline\n\n';
    report += '| Metric | Current | Baseline | Change | Threshold |\n';
    report += '|--------|----------|-----------|---------|------------|\n';
    
    byType.regression.forEach(reg => {
      report += `| ${reg.metric} | ${reg.current.toFixed(2)} | ${reg.baseline.toFixed(2)} | ${reg.change} | ${reg.threshold} |\n`;
    });
    
    report += '\n';
  }
  
  // Report threshold violations
  if (byType.threshold) {
    report += '## Threshold Violations\n\n';
    report += '| Metric | Current | Threshold | Unit |\n';
    report += '|--------|----------|-----------|------|\n';
    
    byType.threshold.forEach(reg => {
      report += `| ${reg.metric} | ${reg.current.toFixed(2)} | ${reg.threshold} | ${reg.unit} |\n`;
    });
  }
  
  return report;
}

// Run checks
const regressions = checkRegressions();
const report = generateReport(regressions);

// Write report
fs.writeFileSync(
  path.join(process.cwd(), 'performance-report.md'),
  report
);

// Exit with error if regressions found
if (regressions.length > 0) {
  console.error(report);
  process.exit(1);
} else {
  console.log(report);
} 