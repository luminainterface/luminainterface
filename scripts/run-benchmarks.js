const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');
const { profiler } = require('../frontend/src/tests/benchmark/performanceMetrics');

// Performance thresholds
const THRESHOLDS = {
  avgLatency: 200, // ms
  peakMemory: 384, // MB
  conceptThroughput: 50, // concepts/s
  renderFps: 45,
  embedLatency: 100, // ms
  searchLatency: 150, // ms
  inferenceLatency: 250 // ms
};

async function runBenchmarks() {
  console.log('Starting performance benchmarks...');
  
  // Launch browser for UI tests
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Run core benchmarks
    const result = await profiler.runFullBenchmark();
    
    // Run additional stress tests
    await runStressTests(page);
    
    // Generate detailed report
    const report = generateReport(result);
    
    // Save results
    fs.writeFileSync(
      path.join(process.cwd(), 'benchmark-results.json'),
      JSON.stringify(result, null, 2)
    );
    
    fs.writeFileSync(
      path.join(process.cwd(), 'benchmark-report.html'),
      report
    );
    
    // Check for regressions
    const warnings = checkThresholds(result);
    if (warnings.length > 0) {
      console.error('Performance regressions detected:');
      console.error(warnings);
      process.exit(1);
    }
    
    console.log('Benchmarks completed successfully');
    
  } catch (error) {
    console.error('Benchmark failed:', error);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

async function runStressTests(page) {
  console.log('Running stress tests...');
  
  // Chat flood test
  await page.goto('http://localhost:3000');
  const messages = generateTestMessages(500); // 100 msg/s for 5s
  
  console.log('Running chat flood test...');
  const chatStart = Date.now();
  
  for (const msg of messages) {
    await page.fill('[data-test="chat-input"]', msg);
    await page.keyboard.press('Enter');
    
    // Maintain 100 msg/s rate
    const elapsed = Date.now() - chatStart;
    const target = (messages.indexOf(msg) + 1) * 10; // 10ms per message
    if (elapsed < target) {
      await new Promise(r => setTimeout(r, target - elapsed));
    }
  }
  
  // Crawl storm test
  console.log('Running crawl storm test...');
  const crawlConcepts = generateCrawlConcepts(1000);
  
  for (const concept of crawlConcepts) {
    await page.click(`[data-test="crawl-${concept}"]`);
  }
  
  // Graph explosion test
  console.log('Running graph explosion test...');
  const nodes = generateGraphNodes(10000);
  await page.evaluate((nodes) => {
    window.testInjectNodes(nodes);
  }, nodes);
  
  // Let the system stabilize
  await new Promise(r => setTimeout(r, 5000));
}

function generateTestMessages(count) {
  const templates = [
    'Tell me about {concept}',
    'How does {concept} work?',
    'Explain the relationship between {concept} and {concept2}',
    'What are best practices for {concept}?',
    'Compare {concept} with {concept2}'
  ];
  
  const concepts = [
    'vector search',
    'neural networks',
    'semantic indexing',
    'concept drift',
    'knowledge graphs',
    'embedding models',
    'attention mechanisms',
    'transformer architecture',
    'clustering algorithms',
    'dimensionality reduction'
  ];
  
  return Array(count).fill(0).map(() => {
    const template = templates[Math.floor(Math.random() * templates.length)];
    const concept = concepts[Math.floor(Math.random() * concepts.length)];
    const concept2 = concepts[Math.floor(Math.random() * concepts.length)];
    return template
      .replace('{concept}', concept)
      .replace('{concept2}', concept2);
  });
}

function generateCrawlConcepts(count) {
  const prefixes = ['Neural', 'Vector', 'Semantic', 'Graph', 'Deep'];
  const suffixes = ['Model', 'Network', 'System', 'Algorithm', 'Architecture'];
  
  return Array(count).fill(0).map(() => {
    const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
    const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
    return `${prefix}_${suffix}_${Math.random().toString(36).slice(2, 5)}`;
  });
}

function generateGraphNodes(count) {
  const nodes = [];
  const edges = [];
  
  for (let i = 0; i < count; i++) {
    nodes.push({
      id: `node_${i}`,
      label: `Concept ${i}`,
      score: Math.random(),
      coverage: Math.random()
    });
    
    // Add some edges (average degree of 3)
    for (let j = 0; j < 3; j++) {
      const target = Math.floor(Math.random() * i);
      if (target !== i) {
        edges.push({
          source: `node_${i}`,
          target: `node_${target}`,
          strength: Math.random()
        });
      }
    }
  }
  
  return { nodes, edges };
}

function checkThresholds(result) {
  const warnings = [];
  
  for (const [metric, threshold] of Object.entries(THRESHOLDS)) {
    const value = result.summary[metric];
    if (value > threshold) {
      warnings.push({
        metric,
        value,
        threshold
      });
    }
  }
  
  return warnings;
}

function generateReport(result) {
  return `
<!DOCTYPE html>
<html>
<head>
  <title>Lumina Performance Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 2rem; }
    .metric-card { background: #f3f4f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    .chart { height: 400px; margin-bottom: 2rem; }
    .warning { color: #ef4444; }
  </style>
</head>
<body>
  <h1>Lumina Performance Report</h1>
  <div id="summary">
    <h2>Summary</h2>
    ${Object.entries(result.summary)
      .map(([metric, value]) => `
        <div class="metric-card">
          <h3>${metric}</h3>
          <p class="${value > THRESHOLDS[metric] ? 'warning' : ''}">${value.toFixed(2)}</p>
        </div>
      `).join('')}
  </div>
  
  <div id="latency-chart" class="chart"></div>
  <div id="memory-chart" class="chart"></div>
  <div id="fps-chart" class="chart"></div>
  
  <script>
    const metrics = ${JSON.stringify(result.metrics)};
    
    function createChart(elementId, data, title) {
      Plotly.newPlot(elementId, [{
        x: data.map(d => new Date(d.timestamp)),
        y: data.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers'
      }], {
        title,
        xaxis: { title: 'Time' },
        yaxis: { title: data[0].unit }
      });
    }
    
    createChart(
      'latency-chart',
      metrics.filter(m => m.unit === 'ms'),
      'Response Latency'
    );
    
    createChart(
      'memory-chart',
      metrics.filter(m => m.unit === 'mb'),
      'Memory Usage'
    );
    
    createChart(
      'fps-chart',
      metrics.filter(m => m.unit === 'fps'),
      'Render Performance'
    );
  </script>
</body>
</html>
  `;
}

runBenchmarks(); 