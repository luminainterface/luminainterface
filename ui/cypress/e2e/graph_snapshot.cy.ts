/// <reference types="cypress" />

describe('FractalView visual snapshot', () => {
  const base = Cypress.env('CYPRESS_BASE_URL') || 'http://localhost:5173'

  it('renders initial circle-packing layout', () => {
    cy.visitGraph()
    cy.waitForApiHealth()
    cy.wait(500) // Additional wait for layout to stabilize
    cy.matchImageSnapshot('fractal-initial')
  })
}) 