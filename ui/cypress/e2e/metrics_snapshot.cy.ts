/// <reference types="cypress" />

describe('Metrics panel visual snapshot', () => {
  beforeEach(() => {
    cy.openMetricsPanel()
    cy.waitForApiHealth()
  })

  it('renders metrics panel with charts', () => {
    // Verify chart data is loaded
    cy.get('.metrics-panel').within(() => {
      cy.get('canvas').should('have.length.at.least', 2)
      cy.get('.chart-title').should('be.visible')
    })
    
    // Take snapshot of the entire panel
    cy.get('.metrics-panel').matchImageSnapshot('metrics-panel-initial')
  })

  it('renders individual charts correctly', () => {
    // Take snapshots of individual charts
    cy.get('.metrics-panel').within(() => {
      cy.get('canvas').each(($canvas, index) => {
        cy.wrap($canvas).matchImageSnapshot(`metrics-chart-${index + 1}`)
      })
    })
  })
}) 