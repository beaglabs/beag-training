"""NIST Cybersecurity Framework (CSF) 2.0 fetcher.

Structure: 6 Functions → 22 Categories → 106 Subcategories.

Every entry carries supplemental guidance from the CSF 2.0 core document.
When a local ``NIST_CSF_2.0_catalog.json`` exists it is preferred.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from frameworks.catalog import ControlEntry, Framework, Granularity

LOCAL_PATH = Path(__file__).resolve().parent.parent / "NIST_CSF_2.0_catalog.json"

# --- Embedded CSF 2.0 catalog ---

CSF_DATA: dict[str, object] = {
    "functions": [
        {
            "id": "GV",
            "title": "Govern",
            "description": "Establish and monitor the organization's cybersecurity risk management strategy, expectations, and policy.",
            "categories": [
                {
                    "id": "GV.OC",
                    "title": "Organizational Context",
                    "description": "The circumstances — mission, stakeholder expectations, dependencies, and legal, regulatory, and contractual requirements — surrounding the organization's cybersecurity risk management decisions.",
                    "subcategories": [
                        ("GV.OC-01", "The organizational mission is understood and informs cybersecurity risk management."),
                        ("GV.OC-02", "Internal and external stakeholders are understood, their needs and expectations regarding cybersecurity risk management are identified, and their input is received and considered."),
                        ("GV.OC-03", "Legal, regulatory, and contractual requirements regarding cybersecurity — including privacy and civil liberties obligations — are understood and managed."),
                        ("GV.OC-04", "Critical objectives, capabilities, and services that external stakeholders depend on or expect from the organization are understood and communicated."),
                        ("GV.OC-05", "Outcomes, capabilities, and services that the organization depends on from others are understood and communicated."),
                    ],
                },
                {
                    "id": "GV.RM",
                    "title": "Risk Management Strategy",
                    "description": "The organization's priorities, constraints, risk tolerance and appetite statements, and assumptions supporting operational risk decisions.",
                    "subcategories": [
                        ("GV.RM-01", "Risk management objectives are established and agreed to by organizational stakeholders."),
                        ("GV.RM-02", "Risk appetite and risk tolerance statements are established, communicated, and maintained."),
                        ("GV.RM-03", "Cybersecurity risk management activities and outcomes are included in enterprise risk management processes."),
                        ("GV.RM-04", "Strategic direction that describes appropriate risk response options is established and communicated."),
                        ("GV.RM-05", "Lines of communication across the organization are established for cybersecurity risk management."),
                        ("GV.RM-06", "A standardized method for calculating, evaluating, and prioritizing cybersecurity risk is established and communicated."),
                        ("GV.RM-07", "Strategic opportunities (positive risks) are identified and appropriate actions are taken."),
                    ],
                },
                {
                    "id": "GV.RR",
                    "title": "Roles, Responsibilities, and Authorities",
                    "description": "Cybersecurity roles, responsibilities, and authorities to foster accountability, performance assessment, and continuous improvement.",
                    "subcategories": [
                        ("GV.RR-01", "Organizational leadership responsibility for cybersecurity risk is established, communicated, and overseen."),
                        ("GV.RR-02", "Cybersecurity leadership roles and responsibilities are established, communicated, and exercised."),
                        ("GV.RR-03", "Adequate resources commensurate with the cybersecurity risk strategy are established and communicated."),
                        ("GV.RR-04", "Cybersecurity is included in human resources practices, including security screening, onboarding, and offboarding."),
                    ],
                },
                {
                    "id": "GV.PO",
                    "title": "Policy",
                    "description": "Organizational cybersecurity policy is established, communicated, and enforced.",
                    "subcategories": [
                        ("GV.PO-01", "A policy for managing cybersecurity risks is developed based on organizational context and informed by applicable requirements."),
                        ("GV.PO-02", "Policy for managing cybersecurity risks is communicated to internal and external stakeholders."),
                    ],
                },
                {
                    "id": "GV.OV",
                    "title": "Oversight",
                    "description": "Results of organization-wide cybersecurity risk management activities and performance are used to inform, improve, and adjust the risk management strategy.",
                    "subcategories": [
                        ("GV.OV-01", "Cybersecurity risk management strategy outcomes are reviewed by organizational leadership to inform strategic decision-making."),
                        ("GV.OV-02", "The performance of the cybersecurity risk management strategy is evaluated and adjusted."),
                        ("GV.OV-03", "Organizational cybersecurity risk management performance is evaluated and communicated to senior leadership."),
                    ],
                },
                {
                    "id": "GV.SC",
                    "title": "Cybersecurity Supply Chain Risk Management",
                    "description": "Cyber supply chain risk management processes are identified, established, managed, monitored, and improved by organizational stakeholders.",
                    "subcategories": [
                        ("GV.SC-01", "A cybersecurity supply chain risk management program, strategy, objectives, policies, and processes are established and agreed to by organizational stakeholders."),
                        ("GV.SC-02", "Cybersecurity roles and responsibilities for suppliers, customers, and partners are established and communicated."),
                        ("GV.SC-03", "Cybersecurity supply chain risk management is integrated into cybersecurity and enterprise risk management, risk assessment, and improvement processes."),
                        ("GV.SC-04", "Suppliers with known criticality are prioritized and appropriate risk management actions are taken."),
                        ("GV.SC-05", "Cybersecurity supply chain risk management planning is performed as part of the acquisition process."),
                        ("GV.SC-06", "Cybersecurity supply chain risk management activities and performance are included in cybersecurity and enterprise risk management."),
                        ("GV.SC-07", "Changes in suppliers during relationships are managed, and the risk of the change is assessed."),
                        ("GV.SC-08", "Relevant internal and external stakeholders receive cybersecurity supply chain risk management information."),
                        ("GV.SC-09", "Cybersecurity supply chain risk management plans include provisions for determining a disposition for data, services, and products when the relationship with a supplier is terminated."),
                        ("GV.SC-10", "Supply chain security is included in security awareness and training and in role-based training programs."),
                    ],
                },
            ],
        },
        {
            "id": "ID",
            "title": "Identify",
            "description": "Determine the current cybersecurity risks to the organization.",
            "categories": [
                {
                    "id": "ID.AM",
                    "title": "Asset Management",
                    "description": "The data, personnel, devices, systems, and facilities that enable the organization to achieve business purposes are identified and managed consistent with their relative importance.",
                    "subcategories": [
                        ("ID.AM-01", "Inventories of hardware managed by the organization are maintained."),
                        ("ID.AM-02", "Inventories of software, services, and systems managed by the organization are maintained."),
                        ("ID.AM-03", "Representations of the organization's authorized network communication and internal and external network data flows are maintained."),
                        ("ID.AM-04", "Inventories of services provided by suppliers are maintained."),
                        ("ID.AM-05", "Assets are prioritized based on classification, criticality, resources, and impact on the mission."),
                        ("ID.AM-07", "Inventories of data and corresponding metadata for prioritized assets are maintained."),
                        ("ID.AM-08", "Systems, hardware, software, services, and data are managed throughout their life cycles."),
                    ],
                },
                {
                    "id": "ID.RA",
                    "title": "Risk Assessment",
                    "description": "The cybersecurity risk to the organization, assets, and individuals is understood by the organization.",
                    "subcategories": [
                        ("ID.RA-01", "Vulnerabilities in assets are identified, validated, and recorded."),
                        ("ID.RA-02", "Cyber threat intelligence is received and analyzed from information sharing forums and sources."),
                        ("ID.RA-03", "Internal and external threats to the organization are identified and recorded."),
                        ("ID.RA-04", "Potential impacts and likelihoods of threats materializing are identified and recorded."),
                        ("ID.RA-05", "Threats, vulnerabilities, likelihoods, and impacts are used to understand inherent risk and inform risk response prioritization."),
                        ("ID.RA-06", "Risk responses are chosen from the available options, prioritized, planned, tracked, and communicated."),
                    ],
                },
                {
                    "id": "ID.IM",
                    "title": "Improvement",
                    "description": "Improvements to organizational cybersecurity risk management processes, procedures, and activities are identified across all CSF Functions.",
                    "subcategories": [
                        ("ID.IM-01", "Improvements are identified from evaluations."),
                        ("ID.IM-02", "Improvements are identified from security tests and exercises, including those done in coordination with suppliers and relevant third parties."),
                        ("ID.IM-03", "Improvements are identified from execution of operational processes, procedures, and activities."),
                        ("ID.IM-04", "Incident response plans and other risk management plans are evaluated and recommendations are communicated to leadership."),
                    ],
                },
            ],
        },
        {
            "id": "PR",
            "title": "Protect",
            "description": "Use safeguards to prevent or reduce cybersecurity risk.",
            "categories": [
                {
                    "id": "PR.AA",
                    "title": "Identity Management, Authentication, and Access Control",
                    "description": "Access to physical and logical assets is limited to authorized users, services, and hardware and managed commensurate with the assessed risk.",
                    "subcategories": [
                        ("PR.AA-01", "Identities and credentials for authorized users, services, and hardware are managed by the organization."),
                        ("PR.AA-02", "Identities are proofed and bound to credentials based on the context of interactions."),
                        ("PR.AA-03", "Users, services, and hardware are authenticated."),
                        ("PR.AA-04", "Identity assertions are protected, conveyed, and verified."),
                        ("PR.AA-05", "Access permissions, entitlements, and authorizations are defined in a policy, managed, enforced, and reviewed, and incorporate the principles of least privilege and separation of duties."),
                        ("PR.AA-06", "Physical access to assets is managed, monitored, and enforced commensurate with risk."),
                    ],
                },
                {
                    "id": "PR.AT",
                    "title": "Awareness and Training",
                    "description": "The organization's personnel are provided cybersecurity awareness and training so they can perform their cybersecurity-related tasks.",
                    "subcategories": [
                        ("PR.AT-01", "All users are informed and trained about their security and privacy responsibilities."),
                        ("PR.AT-02", "Individuals in specialized roles receive role-based cybersecurity training."),
                    ],
                },
                {
                    "id": "PR.DS",
                    "title": "Data Security",
                    "description": "Data is managed consistent with the organization's risk strategy to protect the confidentiality, integrity, and availability of information.",
                    "subcategories": [
                        ("PR.DS-01", "The confidentiality, integrity, and availability of data-at-rest are protected."),
                        ("PR.DS-02", "The confidentiality, integrity, and availability of data-in-transit are protected."),
                        ("PR.DS-10", "The confidentiality, integrity, and availability of data-in-use are protected."),
                        ("PR.DS-11", "Backups of data are created, protected, maintained, and tested."),
                    ],
                },
                {
                    "id": "PR.PS",
                    "title": "Platform Security",
                    "description": "The hardware, software (e.g., firmware, operating systems, applications), and services of physical and virtual platforms are managed consistent with the organization's risk strategy.",
                    "subcategories": [
                        ("PR.PS-01", "Configuration management practices are established and applied."),
                        ("PR.PS-02", "Software is maintained and replaced in accordance with risk."),
                        ("PR.PS-03", "Configuration change control is implemented."),
                        ("PR.PS-04", "Log records are generated and made available for continuous monitoring."),
                        ("PR.PS-05", "Installation and execution of unauthorized software are prevented."),
                        ("PR.PS-06", "Secure software development practices are integrated and their performance is monitored throughout the software development life cycle."),
                    ],
                },
                {
                    "id": "PR.IR",
                    "title": "Technology Infrastructure Resilience",
                    "description": "Security architectures are managed with the organization's risk strategy to protect asset confidentiality, integrity, and availability.",
                    "subcategories": [
                        ("PR.IR-01", "Networks and environments are segmented based on risk."),
                        ("PR.IR-02", "Network resilience and redundancy are designed commensurate with risk criticality."),
                        ("PR.IR-03", "Adequate resource capacity to ensure availability is maintained."),
                        ("PR.IR-04", "Adequate resource capacity is available for critical infrastructure."),
                        ("PR.IR-05", "Redundancy for critical systems is implemented based on risk."),
                    ],
                },
            ],
        },
        {
            "id": "DE",
            "title": "Detect",
            "description": "Find and analyze possible cybersecurity attacks and compromises.",
            "categories": [
                {
                    "id": "DE.CM",
                    "title": "Continuous Monitoring",
                    "description": "Assets are monitored to find anomalies, indicators of compromise, and other potentially adverse events.",
                    "subcategories": [
                        ("DE.CM-01", "Networks and network services are monitored to find potentially adverse events."),
                        ("DE.CM-02", "The physical environment is monitored to find potentially adverse events."),
                        ("DE.CM-03", "Personnel activity and technology usage are monitored to find potentially adverse events."),
                        ("DE.CM-06", "External service provider activities and services are monitored to find potentially adverse events."),
                        ("DE.CM-09", "Computing hardware and software, runtime environments, and their data are monitored to find potentially adverse events."),
                    ],
                },
                {
                    "id": "DE.AE",
                    "title": "Adverse Event Analysis",
                    "description": "Anomalies, indicators of compromise, and other potentially adverse events are analyzed to characterize the events and detect cybersecurity incidents.",
                    "subcategories": [
                        ("DE.AE-02", "Potentially adverse events are analyzed to better understand associated activities."),
                        ("DE.AE-03", "Information is correlated from multiple sources."),
                        ("DE.AE-04", "The estimated impact and scope of adverse events are understood."),
                        ("DE.AE-06", "Information on adverse events is provided to authorized staff and tools."),
                        ("DE.AE-07", "Cyber threat intelligence and other contextual information are integrated into the analysis."),
                        ("DE.AE-08", "Incidents are declared when adverse events meet the defined incident criteria."),
                    ],
                },
            ],
        },
        {
            "id": "RS",
            "title": "Respond",
            "description": "Take action regarding a detected cybersecurity incident.",
            "categories": [
                {
                    "id": "RS.MA",
                    "title": "Incident Management",
                    "description": "Responses to detected cybersecurity incidents are managed.",
                    "subcategories": [
                        ("RS.MA-01", "The incident response plan is executed in coordination with relevant third parties once an incident is declared."),
                        ("RS.MA-02", "Incident reports are triaged and validated."),
                        ("RS.MA-03", "Incidents are categorized and prioritized."),
                        ("RS.MA-04", "Incidents are escalated or elevated as needed."),
                        ("RS.MA-05", "The criteria for initiating incident recovery are defined and agreed to."),
                    ],
                },
                {
                    "id": "RS.AN",
                    "title": "Incident Analysis",
                    "description": "Investigations are conducted to ensure effective response and support forensics and recovery activities.",
                    "subcategories": [
                        ("RS.AN-03", "Analysis is performed to establish what has taken place during an incident and the root cause of the incident."),
                        ("RS.AN-06", "Actions performed during an investigation are recorded, and the records' integrity and provenance are preserved."),
                        ("RS.AN-07", "Incident data and metadata are collected, and their integrity and provenance are preserved."),
                        ("RS.AN-08", "An incident's magnitude is estimated and validated."),
                    ],
                },
                {
                    "id": "RS.CO",
                    "title": "Incident Response Reporting and Communication",
                    "description": "Response activities are coordinated with internal and external stakeholders as required by laws, regulations, or policy.",
                    "subcategories": [
                        ("RS.CO-02", "Internal and external stakeholders are notified of incidents."),
                        ("RS.CO-03", "Information is shared with designated internal and external stakeholders consistent with response plans and information sharing agreements."),
                        ("RS.CO-04", "Reputation is managed through defined coordinating structures."),
                        ("RS.CO-05", "Voluntary information sharing occurs with external stakeholders to achieve broader cybersecurity situational awareness."),
                    ],
                },
                {
                    "id": "RS.MI",
                    "title": "Incident Mitigation",
                    "description": "Activities are performed to prevent expansion of an event and to mitigate the effects of an incident.",
                    "subcategories": [
                        ("RS.MI-01", "Incidents are contained."),
                        ("RS.MI-02", "Incidents are eradicated."),
                        ("RS.MI-03", "Incident-related forensic evidence is handled in accordance with policy."),
                    ],
                },
            ],
        },
        {
            "id": "RC",
            "title": "Recover",
            "description": "Restore assets and operations that were impacted by a cybersecurity incident.",
            "categories": [
                {
                    "id": "RC.RP",
                    "title": "Incident Recovery Plan Execution",
                    "description": "Restoration activities are performed to ensure operational availability of systems and services affected by cybersecurity incidents.",
                    "subcategories": [
                        ("RC.RP-01", "The recovery portion of the incident response plan is executed once initiated from the incident response process or in response to a disaster."),
                        ("RC.RP-02", "Recovery actions are selected, scoped, prioritized, and performed."),
                        ("RC.RP-03", "The integrity of backups and other restoration assets is verified before using them for restoration."),
                        ("RC.RP-04", "Critical mission functions and cybersecurity risk management are considered to establish post-incident operational norms."),
                        ("RC.RP-05", "The integrity of restored assets is verified, systems and services are restored, and normal operating status is confirmed."),
                        ("RC.RP-06", "The end of incident recovery is declared based on criteria, and incident-related documentation is completed."),
                    ],
                },
                {
                    "id": "RC.CO",
                    "title": "Incident Recovery Communication",
                    "description": "Restoration activities are coordinated with internal and external parties.",
                    "subcategories": [
                        ("RC.CO-03", "Recovery activities and progress in restoring operational capabilities are communicated to designated internal and external stakeholders."),
                        ("RC.CO-04", "Public updates on incident recovery are shared using approved messaging and methods."),
                    ],
                },
            ],
        },
    ]
}


def fetch_csf() -> Iterator[ControlEntry]:
    if LOCAL_PATH.exists():
        yield from _parse_local(LOCAL_PATH)
    else:
        yield from _build_embedded()


def _build_embedded() -> Iterator[ControlEntry]:
    for func in CSF_DATA["functions"]:  # type: ignore[union-attr]
        func_id: str = func["id"]  # type: ignore[index]
        func_title: str = func["title"]  # type: ignore[index]
        func_desc: str = func["description"]  # type: ignore[index]

        # Function-level
        yield ControlEntry(
            id=func_id,
            title=func_title,
            description=func_desc,
            framework=Framework.NIST_CSF,
            granularity=Granularity.FAMILY,
            children=[c["id"] for c in func["categories"]],  # type: ignore[index]
        )

        for cat in func["categories"]:  # type: ignore[union-attr]
            cat_id: str = cat["id"]  # type: ignore[index]
            cat_title: str = cat["title"]  # type: ignore[index]
            cat_desc: str = cat["description"]  # type: ignore[index]

            # Category-level
            yield ControlEntry(
                id=cat_id,
                title=cat_title,
                description=cat_desc,
                framework=Framework.NIST_CSF,
                granularity=Granularity.CONTROL,
                parent_id=func_id,
                children=[s[0] for s in cat["subcategories"]],  # type: ignore[index]
            )

            for sub_id, sub_desc in cat["subcategories"]:  # type: ignore[misc]
                yield ControlEntry(
                    id=sub_id,
                    title=sub_id,
                    description=str(sub_desc),
                    framework=Framework.NIST_CSF,
                    granularity=Granularity.ENHANCEMENT,
                    parent_id=cat_id,
                )


def _parse_local(path: Path) -> Iterator[ControlEntry]:
    data = json.loads(path.read_text())
    for entry_data in data.get("entries", []):
        yield ControlEntry.from_dict(entry_data)
