/// <reference types="cypress" />

describe('Smoke Tests', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('loads the main application', () => {
    cy.get('#layout').should('exist')
  })

  it('renders the main content area', () => {
    cy.get('#main').should('exist')
  })

  it('renders the health overlay', () => {
    cy.get('.overlay').should('exist')
  })

  it('renders the header', () => {
    cy.get('nav').should('exist')
    cy.get('nav .logo').should('contain', 'Lumina')
  })

  it('renders the metrics panel', () => {
    cy.get('button').contains('Metrics').click()
    cy.get('.metrics-panel').should('be.visible')
  })

  it('renders the chat panel', () => {
    cy.get('.chat-panel').should('exist')
    cy.get('input[placeholder="Ask MasterChat…"]').should('exist')
  })

  it('has working navigation', () => {
    cy.visit('/')
    cy.get('a[href="/graph"]').click()
    cy.url().should('include', '/graph')
  })

  it('metrics endpoint returns valid JSON', () => {
    const apiUrl = Cypress.env('VITE_API_URL') || 'http://localhost:8201'
    cy.request(`${apiUrl}/metrics/summary`).then((response) => {
      expect(response.status).to.eq(200)
      expect(response.body).to.have.property('nodes').and.be.a('number')
      expect(response.body).to.have.property('edges').and.be.a('number')
      expect(response.body).to.have.property('fractal_dimension').and.be.a('number')
    })
  })

  it('shows health status', () => {
    cy.visit('/')
    cy.get('.overlay').should('be.visible')
  })
}) 