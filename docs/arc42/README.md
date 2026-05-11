# arc42 Documentation

This directory contains architecture documentation for DigitalSreeni Image Annotator following the [arc42 template](https://arc42.org/).

## Documentation Structure

| Section | File | Description |
|---------|------|-------------|
| 1 | [Introduction and Goals](01_introduction_and_goals.md) | Overview, features, quality goals, stakeholders |
| 2 | [Architecture Constraints](02_architecture_constraints.md) | Technical, organizational, platform constraints |
| 3 | Context and Scope | *(Not yet documented)* |
| 4 | Solution Strategy | *(Not yet documented)* |
| 5 | [Building Block View](05_building_block_view.md) | System structure, components, data model |
| 6 | [Runtime View](06_runtime_view.md) | Key scenarios and workflows |
| 7 | Deployment View | *(Not yet documented - desktop app)* |
| 8 | [Cross-cutting Concepts](08_crosscutting_concepts.md) | Coordinate systems, conversions, common patterns |
| 9 | [Architecture Decisions](09_architecture_decisions.md) | Key ADRs and rationale |
| 10 | Quality Requirements | *(Covered in section 1)* |
| 11 | [Risks and Technical Debt](11_risks_and_technical_debt.md) | Known issues, limitations, debt |
| 12 | [Glossary](12_glossary.md) | Terms, acronyms, data structures |

## Quick Navigation

### For Developers
- New to the codebase? Start with [Building Block View](05_building_block_view.md)
- Understanding a workflow? See [Runtime View](06_runtime_view.md)
- Need to know coordinate systems? Check [Cross-cutting Concepts](08_crosscutting_concepts.md)

### For Architects
- Review [Architecture Decisions](09_architecture_decisions.md) for design choices
- Check [Architecture Constraints](02_architecture_constraints.md) for limitations
- See [Risks and Technical Debt](11_risks_and_technical_debt.md) for improvement areas

### For Users
- Overview: [Introduction and Goals](01_introduction_and_goals.md)
- Platform compatibility: [Architecture Constraints](02_architecture_constraints.md#platform-constraints)
- Terminology: [Glossary](12_glossary.md)

## Contributing to Documentation

When making architectural changes:
1. Update relevant arc42 sections
2. Add ADR to [Architecture Decisions](09_architecture_decisions.md) if significant
3. Update [Building Block View](05_building_block_view.md) for structural changes
4. Update [Glossary](12_glossary.md) for new terminology
