/// <reference types="cypress" />
/// <reference types="@simonsmith/cypress-image-snapshot" />

declare namespace Cypress {
  interface Chainable {
    matchImageSnapshot(name?: string, options?: any): Chainable<Element>
  }
} 