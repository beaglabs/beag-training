"""CMMC 2.0 fetcher.

CMMC 2.0 models 14 domains → capabilities → practices.
Each practice maps to one or more 800-53 controls.

Level 1:  17 practices (basic safeguarding of FCI)
Level 2:  110 practices (NIST SP 800-171 based, protecting CUI)
Level 3:  Additional practices based on select 800-53 controls (planned)
"""

from __future__ import annotations

from typing import Iterator

from frameworks.catalog import ControlEntry, CrosswalkRef, Framework, Granularity

# --- Embedded CMMC 2.0 Level 2 structure ---

CMMC_DOMAINS: list[dict[str, object]] = [
    {
        "id": "AC",
        "title": "Access Control",
        "description": "Limit information system access to authorized users, processes acting on behalf of authorized users, and devices.",
        "practices": [
            ("AC.L2-3.1.1", "Limit system access to authorized users, processes acting on behalf of authorized users, and devices (including other systems).", "AC-2, AC-3"),
            ("AC.L2-3.1.2", "Limit system access to the types of transactions and functions that authorized users are permitted to execute.", "AC-3, AC-5, AC-6"),
            ("AC.L2-3.1.3", "Control the flow of CUI in accordance with approved authorizations.", "AC-4"),
            ("AC.L2-3.1.4", "Separate the duties of individuals to reduce the risk of malevolent activity without collusion.", "AC-5"),
            ("AC.L2-3.1.5", "Employ the principle of least privilege, including for specific security functions and privileged accounts.", "AC-6"),
            ("AC.L2-3.1.6", "Use non-privileged accounts or roles when accessing non-security functions.", "AC-6"),
            ("AC.L2-3.1.7", "Prevent non-privileged users from executing privileged functions and capture the execution of such functions in audit logs.", "AC-6, AU-3"),
            ("AC.L2-3.1.8", "Limit unsuccessful logon attempts.", "AC-7"),
            ("AC.L2-3.1.9", "Provide privacy and security notices consistent with applicable CUI rules.", "AC-8"),
            ("AC.L2-3.1.10", "Use session lock with pattern-hiding displays to prevent access and viewing of data after a period of inactivity.", "AC-11"),
            ("AC.L2-3.1.11", "Terminate (automatically) a user session after a defined condition.", "AC-12"),
            ("AC.L2-3.1.12", "Monitor and control remote access sessions.", "AC-17"),
            ("AC.L2-3.1.13", "Employ cryptographic mechanisms to protect the confidentiality of remote access sessions.", "AC-17"),
            ("AC.L2-3.1.14", "Route remote access via managed access control points.", "AC-17"),
            ("AC.L2-3.1.15", "Authorize remote execution of privileged commands and remote access to security-relevant information.", "AC-17"),
            ("AC.L2-3.1.16", "Authorize wireless access prior to allowing such connections.", "AC-18"),
            ("AC.L2-3.1.17", "Protect wireless access using authentication and encryption.", "AC-18"),
            ("AC.L2-3.1.18", "Control connection of mobile devices.", "AC-19"),
            ("AC.L2-3.1.19", "Encrypt CUI on mobile devices and mobile computing platforms.", "AC-19"),
            ("AC.L2-3.1.20", "Verify and control/limit connections to and use of external systems.", "AC-20"),
            ("AC.L2-3.1.21", "Limit use of portable storage devices on external systems.", "AC-20"),
            ("AC.L2-3.1.22", "Control CUI posted or processed on publicly accessible systems.", "AC-22"),
        ],
    },
    {
        "id": "AT",
        "title": "Awareness and Training",
        "description": "Ensure managers, systems administrators, and users are made aware of the security risks associated with their activities and of the applicable policies and procedures.",
        "practices": [
            ("AT.L2-3.2.1", "Ensure that managers, system administrators, and users are made aware of security risks associated with their activities.", "AT-2"),
            ("AT.L2-3.2.2", "Ensure that personnel are trained to carry out their assigned information security-related duties.", "AT-3"),
            ("AT.L2-3.2.3", "Provide security awareness training on recognizing and reporting potential indicators of insider threat.", "AT-2"),
        ],
    },
    {
        "id": "AU",
        "title": "Audit and Accountability",
        "description": "Create, protect, and retain system audit records to enable monitoring, analysis, investigation, and reporting of unlawful or unauthorized system activity.",
        "practices": [
            ("AU.L2-3.3.1", "Create and retain system audit logs and records to enable monitoring, analysis, investigation, and reporting.", "AU-2, AU-3, AU-12"),
            ("AU.L2-3.3.2", "Ensure that the actions of individual system users can be uniquely traced to those users.", "AU-2, AU-3"),
            ("AU.L2-3.3.3", "Review and update logged events.", "AU-2"),
            ("AU.L2-3.3.4", "Alert in the event of an audit logging process failure.", "AU-5"),
            ("AU.L2-3.3.5", "Correlate audit record review, analysis, and reporting processes for investigation and response to indications of unlawful, unauthorized, suspicious, or unusual activity.", "AU-6"),
            ("AU.L2-3.3.6", "Provide audit record reduction and report generation to support on-demand analysis and reporting.", "AU-7"),
            ("AU.L2-3.3.7", "Provide a system capability that compares and synchronizes internal system clocks with an authoritative source.", "AU-8"),
            ("AU.L2-3.3.8", "Protect audit information and audit logging tools from unauthorized access, modification, and deletion.", "AU-9"),
            ("AU.L2-3.3.9", "Limit management of audit logging functionality to a subset of privileged users.", "AU-9"),
        ],
    },
    {
        "id": "CA",
        "title": "Security Assessment",
        "description": "Periodically assess the security controls in organizational systems to determine if the controls are effective.",
        "practices": [
            ("CA.L2-3.12.1", "Periodically assess the security controls to determine if the controls are effective in their application.", "CA-2"),
            ("CA.L2-3.12.2", "Develop and implement plans of action designed to correct deficiencies and reduce or eliminate vulnerabilities.", "CA-5"),
            ("CA.L2-3.12.3", "Monitor security controls on an ongoing basis to ensure the continued effectiveness of the controls.", "CA-7"),
            ("CA.L2-3.12.4", "Develop, document, and periodically update system security plans.", "PL-2"),
        ],
    },
    {
        "id": "CM",
        "title": "Configuration Management",
        "description": "Establish and maintain baseline configurations and inventories of organizational systems throughout the system development life cycle.",
        "practices": [
            ("CM.L2-3.4.1", "Establish and maintain baseline configurations and inventories of organizational systems.", "CM-2, CM-8"),
            ("CM.L2-3.4.2", "Establish and enforce security configuration settings for information technology products employed in organizational systems.", "CM-6"),
            ("CM.L2-3.4.3", "Track, review, approve or disapprove, and log changes to organizational systems.", "CM-3"),
            ("CM.L2-3.4.4", "Analyze the security impact of changes prior to implementation.", "CM-4"),
            ("CM.L2-3.4.5", "Define, document, approve, and enforce physical and logical access restrictions associated with changes to organizational systems.", "CM-5"),
            ("CM.L2-3.4.6", "Employ the principle of least functionality by configuring systems to provide only essential capabilities.", "CM-7"),
            ("CM.L2-3.4.7", "Restrict, disable, or prevent the use of nonessential programs, functions, ports, protocols, and services.", "CM-7"),
            ("CM.L2-3.4.8", "Apply deny-by-exception (blacklisting) policy to prevent the use of unauthorized software or deny-all, permit-by-exception (whitelisting) policy to allow execution of authorized software.", "CM-7"),
            ("CM.L2-3.4.9", "Control and monitor user-installed software.", "CM-11"),
        ],
    },
    {
        "id": "IA",
        "title": "Identification and Authentication",
        "description": "Identify information system users, processes acting on behalf of users, or devices and authenticate those identities before granting access.",
        "practices": [
            ("IA.L2-3.5.1", "Identify system users, processes acting on behalf of users, and devices.", "IA-2, IA-3"),
            ("IA.L2-3.5.2", "Authenticate (or verify) the identities of users, processes, or devices as a prerequisite to allowing access.", "IA-2, IA-3"),
            ("IA.L2-3.5.3", "Use multifactor authentication for local and network access to privileged accounts and for network access to non-privileged accounts.", "IA-2"),
            ("IA.L2-3.5.4", "Employ replay-resistant authentication mechanisms for network access to privileged and non-privileged accounts.", "IA-2"),
            ("IA.L2-3.5.5", "Prevent reuse of identifiers for a defined period.", "IA-4"),
            ("IA.L2-3.5.6", "Disable identifiers after a defined period of inactivity.", "IA-4"),
            ("IA.L2-3.5.7", "Enforce a minimum password complexity and change of characters when new passwords are created.", "IA-5"),
            ("IA.L2-3.5.8", "Prohibit password reuse for a specified number of generations.", "IA-5"),
            ("IA.L2-3.5.9", "Allow temporary password use for system logons with an immediate change to a permanent password.", "IA-5"),
            ("IA.L2-3.5.10", "Store and transmit only cryptographically-protected passwords.", "IA-5"),
            ("IA.L2-3.5.11", "Obscure feedback of authentication information.", "IA-6"),
        ],
    },
    {
        "id": "IR",
        "title": "Incident Response",
        "description": "Establish an operational incident handling capability that includes preparation, detection, analysis, containment, recovery, and user response activities.",
        "practices": [
            ("IR.L2-3.6.1", "Establish an operational incident-handling capability for organizational systems that includes preparation, detection, analysis, containment, recovery, and user response activities.", "IR-4"),
            ("IR.L2-3.6.2", "Track, document, and report incidents to designated officials and/or authorities.", "IR-5, IR-6"),
            ("IR.L2-3.6.3", "Test the organizational incident response capability.", "IR-3"),
        ],
    },
    {
        "id": "MA",
        "title": "Maintenance",
        "description": "Perform maintenance on organizational systems.",
        "practices": [
            ("MA.L2-3.7.1", "Perform maintenance on organizational systems.", "MA-2"),
            ("MA.L2-3.7.2", "Provide controls on the tools, techniques, mechanisms, and personnel used to conduct system maintenance.", "MA-3, MA-5"),
            ("MA.L2-3.7.3", "Ensure equipment removed for off-site maintenance is sanitized of any CUI.", "MA-2"),
            ("MA.L2-3.7.4", "Check media containing diagnostic and test programs for malicious code before use.", "MA-3"),
            ("MA.L2-3.7.5", "Require multifactor authentication to establish nonlocal maintenance sessions via external network connections.", "MA-4"),
            ("MA.L2-3.7.6", "Supervise the maintenance activities of maintenance personnel without required access authorization.", "MA-5"),
        ],
    },
    {
        "id": "MP",
        "title": "Media Protection",
        "description": "Protect system media containing CUI, limiting access to authorized users; sanitize or destroy before disposal or release for reuse.",
        "practices": [
            ("MP.L2-3.8.1", "Protect (i.e., physically control and securely store) system media containing CUI.", "MP-4"),
            ("MP.L2-3.8.2", "Limit access to CUI on system media to authorized users.", "MP-2"),
            ("MP.L2-3.8.3", "Sanitize or destroy system media containing CUI before disposal or release for reuse.", "MP-6"),
            ("MP.L2-3.8.4", "Mark media with necessary CUI markings and distribution limitations.", "MP-3"),
            ("MP.L2-3.8.5", "Control access to media containing CUI and maintain accountability during transport outside controlled areas.", "MP-5"),
            ("MP.L2-3.8.6", "Implement cryptographic mechanisms to protect the confidentiality of CUI stored on digital media during transport.", "MP-5"),
            ("MP.L2-3.8.7", "Control the use of removable media on system components.", "MP-7"),
            ("MP.L2-3.8.8", "Prohibit the use of portable storage devices when such devices have no identifiable owner.", "MP-7"),
            ("MP.L2-3.8.9", "Protect the confidentiality of backup CUI at storage locations.", "CP-9"),
        ],
    },
    {
        "id": "PE",
        "title": "Physical Protection",
        "description": "Limit physical access to organizational systems, equipment, and the respective operating environments to authorized individuals.",
        "practices": [
            ("PE.L2-3.10.1", "Limit physical access to organizational systems, equipment, and the respective operating environments to authorized individuals.", "PE-3"),
            ("PE.L2-3.10.2", "Protect and monitor the physical facility and support infrastructure for organizational systems.", "PE-3, PE-6"),
            ("PE.L2-3.10.3", "Escort visitors and monitor visitor activity.", "PE-7, PE-8"),
            ("PE.L2-3.10.4", "Maintain audit logs of physical access.", "PE-6"),
            ("PE.L2-3.10.5", "Control and manage physical access devices.", "PE-3"),
            ("PE.L2-3.10.6", "Enforce safeguarding measures for CUI at alternate work sites.", "PE-3"),
        ],
    },
    {
        "id": "RA",
        "title": "Risk Assessment",
        "description": "Periodically assess the risk to organizational operations resulting from the operation of organizational systems and processing of CUI.",
        "practices": [
            ("RA.L2-3.11.1", "Periodically assess the risk to organizational operations, assets, and individuals resulting from the operation of systems and processing of CUI.", "RA-3"),
            ("RA.L2-3.11.2", "Scan for vulnerabilities in organizational systems and applications periodically and when new vulnerabilities are identified.", "RA-5"),
            ("RA.L2-3.11.3", "Remediate vulnerabilities in accordance with risk assessments.", "RA-5"),
            ("RA.L2-3.11.4", "Share information obtained from the vulnerability scanning with designated personnel throughout the organization.", "RA-5"),
            ("RA.L2-3.11.5", "Employ a means for evaluating cyber threat intelligence sources.", "RA-3"),
            ("RA.L2-3.11.6", "Establish and maintain a cyber threat hunting capability.", "RA-10"),
            ("RA.L2-3.11.7", "Develop a risk response strategy that includes defined courses of action for known risks.", "RA-7"),
        ],
    },
    {
        "id": "SC",
        "title": "System and Communications Protection",
        "description": "Monitor, control, and protect communications at the external and internal boundaries of organizational systems.",
        "practices": [
            ("SC.L2-3.13.1", "Monitor, control, and protect organizational communications at the external boundaries and key internal boundaries.", "SC-7"),
            ("SC.L2-3.13.2", "Employ architectural designs, software development techniques, and systems engineering principles that promote effective information security.", "SA-8"),
            ("SC.L2-3.13.3", "Separate user functionality from system management functionality.", "SC-2"),
            ("SC.L2-3.13.4", "Prevent unauthorized and unintended information transfer via shared system resources.", "SC-4"),
            ("SC.L2-3.13.5", "Implement subnetworks for publicly accessible system components that are physically or logically separated from internal networks.", "SC-7"),
            ("SC.L2-3.13.6", "Deny network communications traffic by default and allow network communications traffic by exception.", "SC-7"),
            ("SC.L2-3.13.7", "Prevent remote devices from simultaneously establishing non-remote connections and communicating via some other connection.", "SC-7"),
            ("SC.L2-3.13.8", "Implement cryptographic mechanisms to prevent unauthorized disclosure of CUI during transmission unless protected by alternative physical safeguards.", "SC-8"),
            ("SC.L2-3.13.9", "Terminate network connections associated with communications sessions at the end of the sessions or after a defined period of inactivity.", "SC-10"),
            ("SC.L2-3.13.10", "Establish and manage cryptographic keys for cryptography employed in organizational systems.", "SC-12"),
            ("SC.L2-3.13.11", "Employ FIPS-validated cryptography when used to protect the confidentiality of CUI.", "SC-13"),
            ("SC.L2-3.13.12", "Prohibit remote activation of collaborative computing devices and provide indication of devices in use to users present at the device.", "SC-15"),
            ("SC.L2-3.13.13", "Control and monitor the use of mobile code.", "SC-18"),
            ("SC.L2-3.13.14", "Control and monitor the use of Voice over Internet Protocol (VoIP) technologies.", "SC-19"),
            ("SC.L2-3.13.15", "Protect the authenticity of communications sessions.", "SC-23"),
            ("SC.L2-3.13.16", "Protect the confidentiality of CUI at rest.", "SC-28"),
        ],
    },
    {
        "id": "SI",
        "title": "System and Information Integrity",
        "description": "Identify, report, and correct system flaws in a timely manner; protect against malicious code; and monitor system security alerts and advisories.",
        "practices": [
            ("SI.L2-3.14.1", "Identify, report, and correct system flaws in a timely manner.", "SI-2"),
            ("SI.L2-3.14.2", "Provide protection from malicious code at designated locations within organizational systems.", "SI-3"),
            ("SI.L2-3.14.3", "Monitor system security alerts and advisories and take appropriate actions in response.", "SI-5"),
            ("SI.L2-3.14.4", "Update malicious code protection mechanisms when new releases are available.", "SI-3"),
            ("SI.L2-3.14.5", "Perform periodic scans of organizational systems and real-time scans of files from external sources.", "SI-3"),
            ("SI.L2-3.14.6", "Monitor organizational systems for unauthorized personnel, connections, devices, and software.", "SI-4"),
            ("SI.L2-3.14.7", "Employ spam protection mechanisms.", "SI-8"),
        ],
    },
    {
        "id": "PS",
        "title": "Personnel Security",
        "description": "Ensure that individuals occupying positions of responsibility are trustworthy and meet established security criteria.",
        "practices": [
            ("PS.L2-3.9.2", "Ensure that organizational systems containing CUI are protected during and after personnel actions such as terminations and transfers.", "PS-4, PS-5"),
        ],
    },
]


def fetch_cmmc() -> Iterator[ControlEntry]:
    yield from _build_embedded()


def _build_embedded() -> Iterator[ControlEntry]:
    for dom in CMMC_DOMAINS:
        dom_id: str = dom["id"]  # type: ignore[assignment]
        dom_title: str = dom["title"]  # type: ignore[assignment]
        dom_desc: str = dom["description"]  # type: ignore[assignment]
        practices: list = dom["practices"]  # type: ignore[assignment]

        yield ControlEntry(
            id=dom_id,
            title=dom_title,
            description=dom_desc,
            framework=Framework.CMMC,
            granularity=Granularity.FAMILY,
            children=[p[0] for p in practices],
        )

        for prac_id, prac_desc, nist_refs in practices:
            refs = _parse_refs(str(nist_refs))
            entry = ControlEntry(
                id=prac_id,
                title=prac_id,
                description=str(prac_desc),
                framework=Framework.CMMC,
                granularity=Granularity.CONTROL,
                parent_id=dom_id,
                crosswalk=[
                    CrosswalkRef(framework=Framework.NIST_800_53, control_id=r)
                    for r in refs
                ],
            )
            yield entry


def _parse_refs(ref_string: str) -> list[str]:
    """Parse a comma-separated list of 800-53 control IDs."""
    return [r.strip() for r in ref_string.split(",") if r.strip()]
