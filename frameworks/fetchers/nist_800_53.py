"""NIST SP 800-53 Rev 5 fetcher.

Provides the full control hierarchy: 20 families → controls → enhancements.

When a local ``NIST_SP-800-53_rev5_catalog.json`` (OSCAL format) exists in the
``frameworks/`` directory, controls are parsed from it.  Otherwise the embedded
structural catalog with key controls is returned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from frameworks.catalog import ControlEntry, Framework, Granularity

OSCAL_PATH = Path(__file__).resolve().parent.parent / "NIST_SP-800-53_rev5_catalog.json"

# --- Embedded family metadata ---

FAMILIES: list[dict[str, object]] = [
    {
        "id": "AC",
        "title": "Access Control",
        "description": "Limit information system access to authorized users, processes acting on behalf of authorized users, and devices.",
        "controls": [
            ("AC-1", "Policy and Procedures", "Develop, document, and disseminate access control policy and procedures."),
            ("AC-2", "Account Management", "Define and document types of accounts; monitor use; manage creation, modification, disabling, and removal."),
            ("AC-3", "Access Enforcement", "Enforce approved authorizations for logical access."),
            ("AC-4", "Information Flow Enforcement", "Enforce approved authorizations for controlling the flow of information within the system."),
            ("AC-5", "Separation of Duties", "Separate duties of individuals to reduce the risk of malevolent activity without collusion."),
            ("AC-6", "Least Privilege", "Employ the principle of least privilege, allowing only authorized accesses for users."),
            ("AC-7", "Unsuccessful Logon Attempts", "Enforce a limit of consecutive invalid logon attempts by a user."),
            ("AC-8", "System Use Notification", "Display an approved system use notification message before granting access."),
            ("AC-10", "Concurrent Session Control", "Limit the number of concurrent sessions for each system account."),
            ("AC-11", "Device Lock", "Prevent further access by initiating a device lock after a period of inactivity."),
            ("AC-12", "Session Termination", "Automatically terminate a user session after defined conditions."),
            ("AC-14", "Permitted Actions without Identification or Authentication", "Identify user actions that can be performed without identification or authentication."),
            ("AC-17", "Remote Access", "Establish and restrict remote access usage, configuration, and monitoring."),
            ("AC-18", "Wireless Access", "Establish usage restrictions, configuration, and monitoring for wireless access."),
            ("AC-19", "Access Control for Mobile Devices", "Establish usage restrictions, configuration, and monitoring for mobile devices."),
            ("AC-20", "Use of External Information Systems", "Establish terms and conditions for use of external information systems."),
            ("AC-21", "Information Sharing", "Enable authorized information sharing based on defined circumstances."),
            ("AC-22", "Publicly Accessible Content", "Designate individuals authorized to make information publicly accessible."),
        ],
    },
    {
        "id": "AT",
        "title": "Awareness and Training",
        "description": "Ensure that managers and users are made aware of the security risks and applicable requirements.",
        "controls": [
            ("AT-1", "Policy and Procedures", "Develop, document, and disseminate awareness and training policy and procedures."),
            ("AT-2", "Literacy Training and Awareness", "Provide security literacy training to information system users."),
            ("AT-3", "Role-Based Training", "Provide role-based security training to personnel with assigned security roles."),
            ("AT-4", "Training Records", "Document and monitor individual information system security training activities."),
        ],
    },
    {
        "id": "AU",
        "title": "Audit and Accountability",
        "description": "Create, protect, and retain information system audit records to enable monitoring, analysis, and reporting.",
        "controls": [
            ("AU-1", "Policy and Procedures", "Develop, document, and disseminate audit and accountability policy and procedures."),
            ("AU-2", "Event Logging", "Identify the types of events that the system is capable of logging."),
            ("AU-3", "Content of Audit Records", "Ensure that audit records contain sufficient information to establish what events occurred."),
            ("AU-4", "Audit Log Storage Capacity", "Allocate audit log storage capacity to meet organizational requirements."),
            ("AU-5", "Response to Audit Logging Process Failures", "Alert designated officials and take defined actions in the event of an audit processing failure."),
            ("AU-6", "Audit Record Review, Analysis, and Reporting", "Review and analyze system audit records for indications of inappropriate or unusual activity."),
            ("AU-7", "Audit Record Reduction and Report Generation", "Provide audit record reduction and report generation capabilities."),
            ("AU-8", "Time Stamps", "Use internal system clocks to generate time stamps for audit records."),
            ("AU-9", "Protection of Audit Information", "Protect audit information and audit logging tools from unauthorized access."),
            ("AU-10", "Non-Repudiation", "Provide protection against an individual falsely denying having performed a particular action."),
            ("AU-11", "Audit Record Retention", "Retain audit records for a defined period of time consistent with records retention policy."),
            ("AU-12", "Audit Record Generation", "Ensure that the information system generates audit records for defined events."),
            ("AU-16", "Cross-Organizational Audit Logging", "Coordinate audit logging among external organizations for cross-organizational events."),
        ],
    },
    {
        "id": "CA",
        "title": "Assessment, Authorization, and Monitoring",
        "description": "Periodically assess security controls; authorize system operation; continuously monitor.",
        "controls": [
            ("CA-1", "Policy and Procedures", "Develop, document, and disseminate assessment, authorization, and monitoring policy."),
            ("CA-2", "Control Assessments", "Select and execute the appropriate assessors for security control assessments."),
            ("CA-3", "Information Exchange", "Approve and manage the exchange of information between systems."),
            ("CA-5", "Plan of Action and Milestones", "Develop and update a plan of action and milestones for the system."),
            ("CA-6", "Authorization", "Assign a senior official as authorizing official; authorize the system to operate."),
            ("CA-7", "Continuous Monitoring", "Develop a continuous monitoring strategy and implement monitoring."),
            ("CA-8", "Penetration Testing", "Conduct penetration testing on the information system."),
            ("CA-9", "Internal System Connections", "Authorize internal connections of system components."),
        ],
    },
    {
        "id": "CM",
        "title": "Configuration Management",
        "description": "Establish and maintain baseline configurations throughout the system development life cycle.",
        "controls": [
            ("CM-1", "Policy and Procedures", "Develop, document, and disseminate configuration management policy and procedures."),
            ("CM-2", "Baseline Configuration", "Develop, document, and maintain a current baseline configuration."),
            ("CM-3", "Configuration Change Control", "Determine and document the types of changes that are configuration-controlled."),
            ("CM-4", "Impact Analyses", "Analyze changes to the system to determine potential security and privacy impacts."),
            ("CM-5", "Access Restrictions for Change", "Define, document, approve, and enforce physical and logical access restrictions for changes."),
            ("CM-6", "Configuration Settings", "Establish and document configuration settings that reflect the most restrictive mode."),
            ("CM-7", "Least Functionality", "Configure the system to provide only essential capabilities."),
            ("CM-8", "System Component Inventory", "Develop and document an inventory of system components."),
            ("CM-9", "Configuration Management Plan", "Develop, document, and implement a configuration management plan."),
            ("CM-10", "Software Usage Restrictions", "Use software and associated documentation in accordance with contract agreements."),
            ("CM-11", "User-Installed Software", "Establish policies governing the installation of software by users."),
            ("CM-12", "Information Location", "Identify and document the location of information and where processing occurs."),
            ("CM-14", "Signed Components", "Prevent the installation of software, firmware, and hardware components without verifiable signatures."),
        ],
    },
    {
        "id": "CP",
        "title": "Contingency Planning",
        "description": "Establish, maintain, and implement plans for emergency response, backup operations, and post-disaster recovery.",
        "controls": [
            ("CP-1", "Policy and Procedures", "Develop, document, and disseminate contingency planning policy and procedures."),
            ("CP-2", "Contingency Plan", "Develop a contingency plan that provides recovery objectives and restoration priorities."),
            ("CP-3", "Contingency Training", "Provide contingency training to information system users."),
            ("CP-4", "Contingency Plan Testing", "Test the contingency plan to determine its effectiveness."),
            ("CP-6", "Alternate Storage Site", "Establish an alternate storage site including necessary agreements."),
            ("CP-7", "Alternate Processing Site", "Establish an alternate processing site and initiate necessary agreements."),
            ("CP-8", "Telecommunications Services", "Establish alternate telecommunications services including necessary agreements."),
            ("CP-9", "Information System Backup", "Conduct backups of user-level and system-level information."),
            ("CP-10", "Information System Recovery and Reconstitution", "Provide for the recovery and reconstitution of the system to a known state."),
        ],
    },
    {
        "id": "IA",
        "title": "Identification and Authentication",
        "description": "Identify users, processes, and devices and authenticate those identities before granting access.",
        "controls": [
            ("IA-1", "Policy and Procedures", "Develop, document, and disseminate identification and authentication policy."),
            ("IA-2", "Identification and Authentication (Organizational Users)", "Uniquely identify and authenticate organizational users."),
            ("IA-3", "Device Identification and Authentication", "Uniquely identify and authenticate devices before establishing a connection."),
            ("IA-4", "Identifier Management", "Manage system identifiers by receiving authorization and tracking assignment."),
            ("IA-5", "Authenticator Management", "Manage system authenticators including initial content, verification, and refresh."),
            ("IA-6", "Authentication Feedback", "Obscure authentication feedback during the authentication process."),
            ("IA-7", "Cryptographic Module Authentication", "Authenticate to a cryptographic module that meets applicable federal standards."),
            ("IA-8", "Identification and Authentication (Non-Organizational Users)", "Uniquely identify and authenticate non-organizational users."),
            ("IA-11", "Re-Authentication", "Require users to re-authenticate under defined circumstances."),
            ("IA-12", "Identity Proofing", "Identity proof users that require accounts for logical access."),
        ],
    },
    {
        "id": "IR",
        "title": "Incident Response",
        "description": "Establish an operational incident handling capability that includes preparation, detection, analysis, and recovery.",
        "controls": [
            ("IR-1", "Policy and Procedures", "Develop, document, and disseminate incident response policy and procedures."),
            ("IR-2", "Incident Response Training", "Provide incident response training to information system users."),
            ("IR-3", "Incident Response Testing", "Test the incident response capability using exercises and simulations."),
            ("IR-4", "Incident Handling", "Implement an incident handling capability for security incidents."),
            ("IR-5", "Incident Monitoring", "Track and document information system security incidents."),
            ("IR-6", "Incident Reporting", "Require personnel to report suspected security incidents."),
            ("IR-7", "Incident Response Assistance", "Provide an incident response support resource that offers advice and assistance."),
            ("IR-8", "Incident Response Plan", "Develop an incident response plan that provides a roadmap for implementing incident response."),
        ],
    },
    {
        "id": "MA",
        "title": "Maintenance",
        "description": "Perform periodic and timely maintenance of organizational information systems.",
        "controls": [
            ("MA-1", "Policy and Procedures", "Develop, document, and disseminate maintenance policy and procedures."),
            ("MA-2", "Controlled Maintenance", "Schedule, perform, document, and review records of maintenance on system components."),
            ("MA-3", "Maintenance Tools", "Approve, control, and monitor the use of information system maintenance tools."),
            ("MA-4", "Nonlocal Maintenance", "Approve and monitor nonlocal maintenance and diagnostic activities."),
            ("MA-5", "Maintenance Personnel", "Establish a process for maintenance personnel authorization and maintain a list of authorized personnel."),
            ("MA-6", "Timely Maintenance", "Obtain maintenance support and spare parts for key system components within defined time periods."),
        ],
    },
    {
        "id": "MP",
        "title": "Media Protection",
        "description": "Protect information system media, both digital and non-digital; restrict access to authorized individuals.",
        "controls": [
            ("MP-1", "Policy and Procedures", "Develop, document, and disseminate media protection policy and procedures."),
            ("MP-2", "Media Access", "Restrict access to digital and non-digital media to authorized individuals."),
            ("MP-3", "Media Marking", "Mark information system media indicating the distribution limitations and handling caveats."),
            ("MP-4", "Media Storage", "Physically control and securely store digital and non-digital media."),
            ("MP-5", "Media Transport", "Protect and control digital and non-digital media during transport outside of controlled areas."),
            ("MP-6", "Media Sanitization", "Sanitize information system media prior to disposal, release out of organizational control, or release for reuse."),
            ("MP-7", "Media Use", "Restrict or prohibit the use of certain types of media."),
        ],
    },
    {
        "id": "PE",
        "title": "Physical and Environmental Protection",
        "description": "Limit physical access to equipment and the facility while protecting against environmental hazards.",
        "controls": [
            ("PE-1", "Policy and Procedures", "Develop, document, and disseminate physical and environmental protection policy."),
            ("PE-2", "Physical Access Authorizations", "Develop, approve, and maintain a list of individuals with authorized physical access."),
            ("PE-3", "Physical Access Control", "Enforce physical access authorizations; control ingress and egress points."),
            ("PE-4", "Access Control for Transmission", "Control physical access to system distribution and transmission lines."),
            ("PE-5", "Access Control for Output Devices", "Control physical access to output devices to prevent unauthorized access to output."),
            ("PE-6", "Monitoring Physical Access", "Monitor physical access to the facility to detect and respond to physical security incidents."),
            ("PE-7", "Visitor Control", "Control physical access by visitors including escort requirements."),
            ("PE-8", "Visitor Access Records", "Maintain visitor access records to the facility."),
            ("PE-9", "Power Equipment and Cabling", "Protect power equipment and power cabling from damage and destruction."),
            ("PE-10", "Emergency Shutoff", "Provide the capability of shutting off power to the information system in emergency situations."),
            ("PE-11", "Emergency Power", "Provide a short-term uninterruptible power supply to facilitate an orderly shutdown."),
            ("PE-12", "Emergency Lighting", "Employ and maintain automatic emergency lighting for the information system."),
            ("PE-13", "Fire Protection", "Employ and maintain fire detection and suppression devices."),
            ("PE-14", "Environmental Controls", "Maintain temperature and humidity levels within acceptable limits."),
            ("PE-15", "Water Damage Protection", "Protect the information system from damage resulting from water leakage."),
            ("PE-16", "Delivery and Removal", "Authorize, monitor, and control information system-related items entering and exiting the facility."),
        ],
    },
    {
        "id": "PL",
        "title": "Planning",
        "description": "Develop, document, update, and implement security plans and related planning documents.",
        "controls": [
            ("PL-1", "Policy and Procedures", "Develop, document, and disseminate planning policy and procedures."),
            ("PL-2", "System Security and Privacy Plans", "Develop security and privacy plans for the system, including system boundaries."),
            ("PL-4", "Rules of Behavior", "Establish and make readily available to individuals the rules that describe their responsibilities."),
            ("PL-7", "Concept of Operations", "Develop a Concept of Operations (CONOPS) for the system."),
            ("PL-8", "Security and Privacy Architectures", "Develop security and privacy architectures for the system using a defense-in-depth approach."),
            ("PL-9", "Central Management", "Centralize the management of security controls for the information system."),
        ],
    },
    {
        "id": "PM",
        "title": "Program Management",
        "description": "Manage information security at the organizational level; independent of particular systems.",
        "controls": [
            ("PM-1", "Information Security Program Plan", "Develop and disseminate an organization-wide information security program plan."),
            ("PM-2", "Information Security Program Leadership Role", "Appoint a senior agency official to coordinate the security program."),
            ("PM-3", "Information Security Program Resources", "Ensure that capital planning and investment requests include security resources."),
            ("PM-4", "Plan of Action and Milestones Process", "Implement a process for ensuring that plans of action and milestones are remediated."),
            ("PM-5", "System Inventory", "Develop and update an inventory of information systems."),
            ("PM-6", "Measures of Performance", "Develop, monitor, and report on information security measures of performance."),
            ("PM-7", "Enterprise Architecture", "Develop an enterprise architecture with consideration for information security."),
            ("PM-8", "Critical Infrastructure Plan", "Address information security issues in the development and documentation of critical infrastructure plans."),
            ("PM-9", "Risk Management Strategy", "Develop a comprehensive strategy to manage risk to organizational operations and assets."),
            ("PM-10", "Authorization Process", "Manage the security and privacy state of the system through authorization processes."),
            ("PM-11", "Mission/Business Process Definition", "Define mission and business processes with consideration for information security."),
            ("PM-13", "Security and Privacy Workforce", "Establish a security and privacy workforce development and improvement program."),
            ("PM-14", "Testing, Training, and Monitoring", "Implement testing, training, and monitoring plan for security controls."),
            ("PM-15", "Security and Privacy Groups and Associations", "Establish and institutionalize contact with groups and associations within the security and privacy fields."),
            ("PM-16", "Threat Awareness Program", "Implement a threat awareness program that includes cross-organization information sharing."),
            ("PM-17", "Protecting Controlled Unclassified Information on External Systems", "Protect controlled unclassified information on external systems."),
            ("PM-19", "Privacy Program Plan", "Develop and implement an organization-wide privacy program plan."),
            ("PM-20", "Dissemination of Privacy Program Information", "Maintain a central repository of privacy program information."),
            ("PM-21", "Accounting of Disclosures", "Develop and maintain an accounting of disclosures of personally identifiable information."),
            ("PM-22", "Personally Identifiable Information Quality Management", "Develop and implement policies to improve the quality of PII."),
            ("PM-25", "Minimization of Personally Identifiable Information Used in Testing", "Minimize the use of PII for testing, training, and research."),
            ("PM-27", "Privacy Reporting", "Develop and maintain internal privacy reports that are shared with senior management."),
            ("PM-28", "Risk Framing", "Identify and document assumptions affecting risk assessments and risk responses."),
            ("PM-29", "Risk Management Program Leadership Roles", "Assign and maintain assigned roles for risk management."),
            ("PM-30", "Supply Chain Risk Management Strategy", "Develop an organization-wide strategy for managing supply chain risks."),
            ("PM-31", "Continuous Monitoring Strategy", "Develop an organization-wide strategy for continuous monitoring."),
            ("PM-32", "Purple Team Testing", "Employ the results of purple team exercises to improve security posture."),
        ],
    },
    {
        "id": "PS",
        "title": "Personnel Security",
        "description": "Ensure personnel occupying positions of responsibility are trustworthy and meet security criteria.",
        "controls": [
            ("PS-1", "Policy and Procedures", "Develop, document, and disseminate personnel security policy and procedures."),
            ("PS-2", "Position Risk Designation", "Assign a risk designation to all organizational positions."),
            ("PS-3", "Personnel Screening", "Screen individuals prior to authorizing access to the information system."),
            ("PS-4", "Personnel Termination", "Upon termination of individual employment, terminate information system access."),
            ("PS-5", "Personnel Transfer", "Review and confirm ongoing operational need for logical and physical access authorizations."),
            ("PS-6", "Access Agreements", "Develop and document access agreements for organizational information systems."),
            ("PS-7", "External Personnel Security", "Ensure external providers meet the same personnel security requirements as organizational personnel."),
            ("PS-8", "Personnel Sanctions", "Employ a formal sanctions process for individuals failing to comply with security policies."),
            ("PS-9", "Position Descriptions", "Incorporate security and privacy roles and responsibilities into organizational position descriptions."),
        ],
    },
    {
        "id": "PT",
        "title": "PII Processing and Transparency",
        "description": "Manage the processing of personally identifiable information (PII) with transparency and accountability.",
        "controls": [
            ("PT-1", "Policy and Procedures", "Develop, document, and disseminate PII processing and transparency policy."),
            ("PT-2", "Authority to Process PII", "Document the authority to process personally identifiable information."),
            ("PT-3", "PII Processing Purposes", "Identify the purpose(s) for which PII is collected, used, maintained, and shared."),
            ("PT-4", "Consent", "Obtain consent from individuals prior to any new use or sharing of their PII."),
            ("PT-5", "Privacy Notice", "Provide a privacy notice that is publicly available and describes PII processing activities."),
            ("PT-6", "System of Records Notice", "For systems containing PII processed on behalf of a federal agency, publish SORNs."),
            ("PT-7", "Specific Categories of PII", "Apply specific conditions or protections for special categories of PII."),
        ],
    },
    {
        "id": "RA",
        "title": "Risk Assessment",
        "description": "Assess the risk to organizational operations, assets, and individuals from operation of information systems.",
        "controls": [
            ("RA-1", "Policy and Procedures", "Develop, document, and disseminate risk assessment policy and procedures."),
            ("RA-2", "Security Categorization", "Categorize information and the information system in accordance with applicable laws."),
            ("RA-3", "Risk Assessment", "Conduct a risk assessment, including an assessment of the likelihood and magnitude of harm."),
            ("RA-5", "Vulnerability Monitoring and Scanning", "Monitor and scan for vulnerabilities in the information system and hosted applications."),
            ("RA-7", "Risk Response", "Respond to findings from security and privacy assessments, audits, and continuous monitoring."),
            ("RA-9", "Criticality Analysis", "Identify critical system components and functions by performing a criticality analysis."),
            ("RA-10", "Threat Hunting", "Establish a threat hunting capability based on analysis of threat intelligence and system data."),
        ],
    },
    {
        "id": "SA",
        "title": "System and Services Acquisition",
        "description": "Allocate resources for the acquisition and deployment of information systems and services.",
        "controls": [
            ("SA-1", "Policy and Procedures", "Develop, document, and disseminate system and services acquisition policy."),
            ("SA-2", "Allocation of Resources", "Determine, document, and allocate resources for protecting the system."),
            ("SA-3", "System Development Life Cycle", "Manage the information system using a system development life cycle methodology."),
            ("SA-4", "Acquisition Process", "Include security and privacy requirements in the acquisition contract for the system."),
            ("SA-5", "System Documentation", "Obtain administrator documentation and user documentation for the information system."),
            ("SA-8", "Security and Privacy Engineering Principles", "Apply information security and privacy engineering principles in the specification, design, development, and implementation of information systems."),
            ("SA-9", "External System Services", "Require that providers of external system services comply with organizational security requirements."),
            ("SA-10", "Developer Configuration Management", "Require the developer to perform configuration management during system development."),
            ("SA-11", "Developer Testing and Evaluation", "Require the developer to create and implement a security testing and evaluation plan."),
            ("SA-15", "Development Process, Standards, and Tools", "Require the developer to follow a documented development process."),
            ("SA-22", "Unsupported System Components", "Replace information system components when support is no longer available."),
        ],
    },
    {
        "id": "SC",
        "title": "System and Communications Protection",
        "description": "Protect the confidentiality, integrity, and availability of information at rest, in transit, and in use.",
        "controls": [
            ("SC-1", "Policy and Procedures", "Develop, document, and disseminate system and communications protection policy."),
            ("SC-2", "Separation of System and User Functionality", "Separate user functionality from system management functionality."),
            ("SC-3", "Security Function Isolation", "Isolate security functions from non-security functions."),
            ("SC-4", "Information in Shared System Resources", "Prevent unauthorized and unintended information transfer via shared system resources."),
            ("SC-5", "Denial-of-Service Protection", "Protect against or limit the effects of denial-of-service attacks."),
            ("SC-7", "Boundary Protection", "Monitor and control communications at the external boundary of the system."),
            ("SC-8", "Transmission Confidentiality and Integrity", "Protect the confidentiality and integrity of transmitted information."),
            ("SC-10", "Network Disconnect", "Terminate the network connection associated with a communications session at the end of the session."),
            ("SC-12", "Cryptographic Key Establishment and Management", "Establish and manage cryptographic keys for required cryptography."),
            ("SC-13", "Cryptographic Protection", "Determine and document the required cryptographic protection for the information system."),
            ("SC-15", "Collaborative Computing Devices and Applications", "Prohibit remote activation of collaborative computing devices and applications."),
            ("SC-17", "Public Key Infrastructure Certificates", "Issue public key certificates or obtain PKI certificates from an approved service provider."),
            ("SC-18", "Mobile Code", "Define acceptable and unacceptable mobile code and mobile code technologies."),
            ("SC-20", "Secure Name/Address Resolution Service (Authoritative Source)", "Provide additional data origin authentication and integrity verification artifacts."),
            ("SC-21", "Secure Name/Address Resolution Service (Recursive or Caching Resolver)", "Request and perform data origin authentication and data integrity verification."),
            ("SC-22", "Architecture and Provisioning for Name/Address Resolution Service", "Ensure name/address resolution services are fault-tolerant and implement security controls."),
            ("SC-23", "Session Authenticity", "Protect the authenticity of communications sessions."),
            ("SC-28", "Protection of Information at Rest", "Protect the confidentiality and integrity of information at rest."),
            ("SC-29", "Heterogeneity", "Employ a diverse set of information technologies to limit exploitation risk."),
            ("SC-31", "Covert Channel Analysis", "Perform covert channel analysis to identify and reduce information leakage."),
            ("SC-32", "Information System Partitioning", "Partition the information system into components residing in separate physical domains or environments."),
            ("SC-34", "Non-Modifiable Executable Programs", "Load and execute the operating environment from hardware-enforced, non-modifiable storage."),
            ("SC-36", "Distributed Processing and Storage", "Distribute processing and storage across multiple physical locations."),
            ("SC-38", "Operations Security", "Employ operations security to protect key organizational information."),
            ("SC-39", "Process Isolation", "Maintain a separate execution domain for each executing process."),
            ("SC-40", "Wireless Link Protection", "Protect external and internal wireless links from signal parameter attacks."),
            ("SC-42", "Sensor Capability and Data", "Prohibit the remote activation or use of environmental sensing capabilities."),
            ("SC-43", "Usage Restrictions", "Establish usage restrictions for system components that have no wireless capabilities."),
            ("SC-45", "System Time Synchronization", "Synchronize system clocks within and between systems and with authoritative time sources."),
            ("SC-46", "Cross-Domain Policy Enforcement", "Enforce policy on data to be transferred across security domains."),
            ("SC-47", "Alternate Communications Paths", "Provide alternate communications paths that are physically separate."),
            ("SC-49", "Hardware-Enforced Separation and Policy Enforcement", "Implement hardware-enforced separation and policy enforcement mechanisms."),
            ("SC-50", "Software-Enforced Separation and Policy Enforcement", "Implement software-enforced separation and policy enforcement mechanisms."),
            ("SC-51", "Hardware-Based Protection", "Use hardware-based protection techniques to protect the system from unauthorized access or modification."),
        ],
    },
    {
        "id": "SI",
        "title": "System and Information Integrity",
        "description": "Identify, report, and correct information system flaws in a timely manner; protect the integrity of information.",
        "controls": [
            ("SI-1", "Policy and Procedures", "Develop, document, and disseminate system and information integrity policy."),
            ("SI-2", "Flaw Remediation", "Identify, report, and correct information system flaws."),
            ("SI-3", "Malicious Code Protection", "Implement malicious code protection mechanisms at system entry and exit points."),
            ("SI-4", "System Monitoring", "Monitor the information system to detect attacks and indicators of potential attacks."),
            ("SI-5", "Security Alerts, Advisories, and Directives", "Receive information system security alerts, advisories, and directives on an ongoing basis."),
            ("SI-6", "Security and Privacy Function Verification", "Verify the correct operation of security and privacy functions."),
            ("SI-7", "Software, Firmware, and Information Integrity", "Employ integrity verification mechanisms to detect unauthorized changes."),
            ("SI-8", "Spam Protection", "Employ spam protection mechanisms at system entry and exit points."),
            ("SI-10", "Information Input Validation", "Check the validity of information inputs."),
            ("SI-11", "Error Handling", "Generate error messages that provide information necessary for corrective actions."),
            ("SI-12", "Information Management and Retention", "Manage and retain information within the system."),
            ("SI-13", "Predictable Failure Prevention", "Employ a means to prevent predictable failures of system components."),
            ("SI-14", "Non-Persistence", "Implement the capability to implement non-persistent information system components and services."),
            ("SI-16", "Memory Protection", "Implement protections to prevent unauthorized code execution."),
            ("SI-17", "Fail-Safe Procedures", "Implement procedures to ensure the system fails securely when a failure condition is detected."),
        ],
    },
    {
        "id": "SR",
        "title": "Supply Chain Risk Management",
        "description": "Manage the risks associated with the products and services that information systems rely on.",
        "controls": [
            ("SR-1", "Policy and Procedures", "Develop, document, and disseminate supply chain risk management policy."),
            ("SR-2", "Supply Chain Risk Management Plan", "Develop a plan for managing supply chain risks associated with the IT supply chain."),
            ("SR-3", "Supply Chain Controls and Processes", "Establish a process to identify and address weaknesses in the supply chain."),
            ("SR-4", "Provenance", "Document and monitor the provenance of system components."),
            ("SR-5", "Acquisition Strategies, Tools, and Methods", "Employ acquisition strategies that protect against, mitigate, and manage supply chain risk."),
            ("SR-6", "Supplier Assessments and Reviews", "Assess and review the supply chain-related risks associated with suppliers."),
            ("SR-7", "Supply Chain Operations Security", "Employ operations security to protect supply chain-related information."),
            ("SR-8", "Notification Agreements", "Establish agreements and procedures with entities involved in the supply chain."),
            ("SR-9", "Tamper Resistance and Detection", "Employ tamper protection for system components during the system development life cycle."),
            ("SR-10", "Inspection of Systems or Components", "Inspect information systems and components to detect tampering."),
            ("SR-11", "Component Authenticity", "Develop and implement anti-counterfeit policy and procedures."),
            ("SR-12", "Component Disposal", "Dispose of system components using approved techniques and methods."),
        ],
    },
]


def fetch_800_53() -> Iterator[ControlEntry]:
    """Yield all 800-53 entries: families, controls, enhancements."""
    if OSCAL_PATH.exists():
        yield from _parse_oscal(OSCAL_PATH)
    else:
        yield from _build_embedded()


def _build_embedded() -> Iterator[ControlEntry]:
    for fam in FAMILIES:
        fam_id: str = fam["id"]  # type: ignore[assignment]
        fam_title: str = fam["title"]  # type: ignore[assignment]
        fam_desc: str = fam["description"]  # type: ignore[assignment]

        # Family-level entry
        yield ControlEntry(
            id=fam_id,
            title=fam_title,
            description=fam_desc,
            framework=Framework.NIST_800_53,
            granularity=Granularity.FAMILY,
            children=[c[0] for c in fam["controls"]],  # type: ignore[index]
        )

        for ctrl_id, ctrl_title, ctrl_desc in fam["controls"]:  # type: ignore[misc]
            yield ControlEntry(
                id=ctrl_id,
                title=ctrl_title,
                description=ctrl_desc,
                framework=Framework.NIST_800_53,
                granularity=Granularity.CONTROL,
                parent_id=fam_id,
            )


def _parse_oscal(path: Path) -> Iterator[ControlEntry]:
    """Parse controls from a NIST OSCAL 800-53 Rev 5 catalog JSON."""
    data = json.loads(path.read_text())
    catalog = data.get("catalog", data)

    groups = catalog.get("groups", [])
    if not groups:
        return

    for group in groups:
        fam_id = group.get("id", "")
        fam_title = group.get("title", "")

        yield ControlEntry(
            id=fam_id,
            title=fam_title,
            description="",
            framework=Framework.NIST_800_53,
            granularity=Granularity.FAMILY,
            children=[],
        )

        children: list[str] = []
        for ctrl in group.get("controls", []):
            ctrl_id = ctrl.get("id", "")
            parts = [p.get("label", "") for p in ctrl.get("props", []) if p.get("name") == "label"]
            ctrl_title = parts[0] if parts else ctrl.get("title", "")
            parts_list = ctrl.get("parts", [])
            desc_parts = [p for p in parts_list if p.get("name") == "statement"]
            ctrl_desc = _render_oscal_parts(desc_parts) if desc_parts else ""

            children.append(ctrl_id)

            yield ControlEntry(
                id=ctrl_id,
                title=ctrl_title,
                description=ctrl_desc,
                framework=Framework.NIST_800_53,
                granularity=Granularity.CONTROL,
                parent_id=fam_id,
            )

            # Process enhancements
            for enh in ctrl.get("controls", []):
                enh_id = enh.get("id", "")
                enh_title = enh.get("title", "")
                enh_parts = [p for p in enh.get("parts", []) if p.get("name") == "statement"]
                enh_desc = _render_oscal_parts(enh_parts) if enh_parts else ""

                yield ControlEntry(
                    id=enh_id,
                    title=enh_title,
                    description=enh_desc,
                    framework=Framework.NIST_800_53,
                    granularity=Granularity.ENHANCEMENT,
                    parent_id=ctrl_id,
                )

        # Fixup children on the family entry
        fam_entry = ControlEntry(
            id=fam_id,
            title=fam_title,
            description="",
            framework=Framework.NIST_800_53,
            granularity=Granularity.FAMILY,
            children=children,
        )
        yield fam_entry


def _render_oscal_parts(parts: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for part in parts:
        prose = part.get("prose", "")
        if prose:
            lines.append(str(prose))
        for sub in part.get("parts", []):
            sub_prose = sub.get("prose", "")
            if sub_prose:
                lines.append(str(sub_prose))
    return "\n".join(lines)
