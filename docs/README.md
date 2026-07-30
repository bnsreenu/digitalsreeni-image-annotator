# Documentation

Architecture documentation for the DigitalSreeni Image Annotator project, following the [arc42 template](https://arc42.org/).

## Documentation Index

| # | Section | Description |
|---|---------|-------------|
| 1 | [Introduction and Goals](01_introduction_and_goals.md) | Overview, features, quality goals, stakeholders |
| 2 | [Architecture Constraints](02_architecture_constraints.md) | Technical, organizational, platform constraints |
| 3 | Context and Scope | *(Not yet documented)* |
| 4 | Solution Strategy | *(Not yet documented)* |
| 5 | [Building Block View](05_building_block_view.md) | System structure, components, data model |
| 6 | [Runtime View](06_runtime_view.md) | Key scenarios and workflows |
| 7 | [Deployment View](07_deployment_view.md) | Entry points, the Qt-free boundary, CLI usage |
| 8 | [Cross-cutting Concepts](08_crosscutting_concepts.md) | Coordinate systems, conversions, common patterns |
| 9 | [Architecture Decisions](09_architecture_decisions.md) | Key ADRs and rationale |
| 10 | Quality Requirements | *(Covered in section 1)* |
| 11 | [Risks and Technical Debt](11_risks_and_technical_debt.md) | Known issues, limitations, debt |
| 12 | [Glossary](12_glossary.md) | Terms, acronyms, data structures |

## Quick Navigation

**For Developers:**
- New to the codebase? Start with [Building Block View](05_building_block_view.md)
- Understanding a workflow? See [Runtime View](06_runtime_view.md)
- Need to know coordinate systems? Check [Cross-cutting Concepts](08_crosscutting_concepts.md)

**For Architects:**
- Design rationale: [Architecture Decisions](09_architecture_decisions.md)
- Constraints: [Architecture Constraints](02_architecture_constraints.md)
- Technical debt: [Risks and Technical Debt](11_risks_and_technical_debt.md)

**For Everyone:**
- Project overview: [Introduction and Goals](01_introduction_and_goals.md)
- Platform compatibility: [Architecture Constraints](02_architecture_constraints.md#platform-constraints)
- Terminology: [Glossary](12_glossary.md)

## Other Documentation

- **[CLAUDE.md](../CLAUDE.md)** — Quick reference guide for Claude Code (in repository root)
- **[README.md](../README.md)** — User-facing documentation (in repository root)

## Contributing to Documentation

When making changes to the codebase:

1. **Update arc42 docs** when architecture changes:
   - Add/update ADRs in [Architecture Decisions](09_architecture_decisions.md)
   - Update component descriptions in [Building Block View](05_building_block_view.md)
   - Document new workflows in [Runtime View](06_runtime_view.md)
   - Add new terms to [Glossary](12_glossary.md)

2. **Keep CLAUDE.md lean** — only quick reference info; point here for details.

3. **Document constraints** — add platform issues and known bugs to [Risks and Technical Debt](11_risks_and_technical_debt.md).

4. **Explain decisions** — significant architectural choices deserve ADRs in section 9.
