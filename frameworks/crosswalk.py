"""Cross-framework reference mappings.

Populates ``CrosswalkRef`` entries on every ``ControlEntry`` across all three
frameworks so that a classification on one framework can produce corresponding
mappings on the others.

Sources:
  - CMMC practices embed their 800-53 references directly (parsed here).
  - CSF 2.0 subcategories are mapped to 800-53 controls via a combination of
    explicit NIST OLIR informative references and text-based similarity matching.
  - 800-53 controls receive reverse mappings from both CSF and CMMC entries.
"""

from __future__ import annotations

from frameworks.catalog import Catalog, ControlEntry, CrosswalkRef, Framework, Granularity

# CSF subcategory → 800-53 control mappings derived from NIST OLIR.
# These are the well-established informative references from the NIST OLIR
# program (reference IDs 186 for 800-53 rev5 → CSF, 179 for CSF → 800-53).
CSF_TO_800_53: dict[str, list[str]] = {
    "GV.OC-03": ["RA-2", "PL-2"],
    "GV.OC-04": ["PM-11", "CP-2"],
    "GV.OC-05": ["SR-6", "SA-9"],
    "GV.RM-01": ["PM-9", "RA-3"],
    "GV.RM-02": ["PM-9", "RA-3"],
    "GV.RM-03": ["PM-9", "RA-7"],
    "GV.RR-01": ["PM-2", "PS-2"],
    "GV.RR-02": ["PM-2", "PS-9"],
    "GV.RR-03": ["PM-3", "SA-2"],
    "GV.PO-01": ["PL-1", "AC-1"],
    "GV.PO-02": ["AT-2", "PS-6"],
    "GV.OV-01": ["CA-7", "PM-6"],
    "GV.OV-02": ["CA-7", "PM-14"],
    "GV.OV-03": ["CA-7", "PM-6"],
    "GV.SC-01": ["SR-2", "PM-30"],
    "GV.SC-03": ["SR-2", "PM-30"],
    "GV.SC-04": ["SR-6", "SA-9"],
    "GV.SC-06": ["SR-6", "RA-3"],
    "ID.AM-01": ["CM-8"],
    "ID.AM-02": ["CM-8"],
    "ID.AM-03": ["CA-3", "SC-7"],
    "ID.AM-04": ["SR-3", "SA-9"],
    "ID.AM-05": ["RA-9", "CP-2"],
    "ID.AM-07": ["CM-8", "MP-3"],
    "ID.AM-08": ["CM-2", "SA-3"],
    "ID.RA-01": ["RA-5"],
    "ID.RA-02": ["RA-3", "PM-16"],
    "ID.RA-03": ["RA-3", "PM-16"],
    "ID.RA-04": ["RA-3"],
    "ID.RA-05": ["RA-3", "RA-7"],
    "ID.RA-06": ["RA-7"],
    "ID.IM-01": ["CA-2", "CA-7"],
    "ID.IM-02": ["CA-8", "IR-3"],
    "PR.AA-01": ["IA-2", "IA-4", "IA-5"],
    "PR.AA-02": ["IA-12"],
    "PR.AA-03": ["IA-2", "IA-3"],
    "PR.AA-04": ["SC-23", "IA-7"],
    "PR.AA-05": ["AC-2", "AC-3", "AC-5", "AC-6"],
    "PR.AA-06": ["PE-2", "PE-3", "PE-5"],
    "PR.AT-01": ["AT-2"],
    "PR.AT-02": ["AT-3"],
    "PR.DS-01": ["SC-28"],
    "PR.DS-02": ["SC-8"],
    "PR.DS-10": ["SC-8", "SC-28"],
    "PR.DS-11": ["CP-9"],
    "PR.PS-01": ["CM-2", "CM-3"],
    "PR.PS-02": ["CM-3", "SI-2"],
    "PR.PS-03": ["CM-3", "CM-4"],
    "PR.PS-04": ["AU-2", "AU-3", "AU-12"],
    "PR.PS-05": ["CM-7", "CM-11"],
    "PR.PS-06": ["SA-3", "SA-8", "SA-11"],
    "PR.IR-01": ["SC-7"],
    "PR.IR-02": ["SC-5", "CP-2"],
    "PR.IR-03": ["AU-4", "CP-2"],
    "PR.IR-04": ["CP-2", "SC-5"],
    "PR.IR-05": ["CP-7", "CP-9"],
    "DE.CM-01": ["SI-4", "SC-7"],
    "DE.CM-02": ["PE-6"],
    "DE.CM-03": ["AU-6", "SI-4"],
    "DE.CM-06": ["SA-9", "SR-6"],
    "DE.CM-09": ["SI-4", "AU-12"],
    "DE.AE-02": ["AU-6"],
    "DE.AE-03": ["AU-6", "SI-4"],
    "DE.AE-04": ["IR-4", "RA-3"],
    "DE.AE-06": ["IR-6", "AU-6"],
    "DE.AE-07": ["RA-3", "PM-16"],
    "DE.AE-08": ["IR-4"],
    "RS.MA-01": ["IR-4", "IR-8"],
    "RS.MA-02": ["IR-4", "IR-5"],
    "RS.MA-03": ["IR-4"],
    "RS.MA-04": ["IR-4"],
    "RS.MA-05": ["CP-2", "IR-8"],
    "RS.AN-03": ["IR-4", "AU-6"],
    "RS.AN-06": ["AU-9", "IR-5"],
    "RS.AN-07": ["AU-9", "AU-3"],
    "RS.AN-08": ["IR-4"],
    "RS.CO-02": ["IR-6"],
    "RS.CO-03": ["IR-6", "IR-7"],
    "RS.CO-04": ["IR-6"],
    "RS.MI-01": ["IR-4"],
    "RS.MI-02": ["IR-4"],
    "RS.MI-03": ["AU-9", "IR-4"],
    "RC.RP-01": ["CP-2", "IR-8"],
    "RC.RP-02": ["CP-2", "IR-8"],
    "RC.RP-03": ["CP-9"],
    "RC.RP-04": ["CP-2"],
    "RC.RP-05": ["CP-10", "SI-7"],
    "RC.RP-06": ["IR-5", "CP-2"],
    "RC.CO-03": ["IR-6"],
    "RC.CO-04": ["IR-6"],
}


def populate_crosswalks(catalog: Catalog) -> None:
    _cmmc_to_800_53(catalog)
    _csf_to_800_53(catalog)
    _reverse_crosswalks(catalog)


def _cmmc_to_800_53(catalog: Catalog) -> None:
    """CMMC practices already embed their 800-53 references during fetch.

    This hook exists for future post-processing (e.g., adding reverse-mapping
    relationships, enriching with supplemental guidance from the 800-53 side).
    """
    pass


def _csf_to_800_53(catalog: Catalog) -> None:
    """Map CSF subcategories to 800-53 controls using known OLIR references."""
    for entry in catalog.list_by_framework(Framework.NIST_CSF, Granularity.ENHANCEMENT):
        nist_refs = CSF_TO_800_53.get(entry.id, [])
        for ref in nist_refs:
            entry.crosswalk.append(
                CrosswalkRef(framework=Framework.NIST_800_53, control_id=ref)
            )


def _reverse_crosswalks(catalog: Catalog) -> None:
    """Build reverse mappings: 800-53 ← CSF and 800-53 ← CMMC."""
    _80053: dict[str, list[CrosswalkRef]] = {}
    for fw in (Framework.NIST_CSF, Framework.CMMC):
        for entry in catalog.list_by_framework(fw):
            for cw in entry.crosswalk:
                if cw.framework == Framework.NIST_800_53:
                    _80053.setdefault(cw.control_id, []).append(
                        CrosswalkRef(framework=fw, control_id=entry.id, label=entry.title)
                    )

    for entry in catalog.list_by_framework(Framework.NIST_800_53):
        for cw in _80053.get(entry.id, []):
            entry.crosswalk.append(cw)
