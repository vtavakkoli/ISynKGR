from __future__ import annotations

from pathlib import Path


def make_nodeset(i: int) -> str:
    return f'''<?xml version="1.0" encoding="utf-8"?>\n<UANodeSet xmlns="http://opcfoundation.org/UA/2011/03/UANodeSet.xsd">\n  <UAObjectType NodeId="ns=2;i={1000+i}" BrowseName="2:Pump{i}">\n    <DisplayName>Pump{i}</DisplayName>\n    <References><Reference ReferenceType="HasComponent">ns=2;i={2000+i}</Reference></References>\n  </UAObjectType>\n  <UAVariable NodeId="ns=2;i={2000+i}" BrowseName="2:Pressure{i}">\n    <DisplayName>Pressure{i}</DisplayName>\n  </UAVariable>\n</UANodeSet>'''


def main() -> None:
    root = Path("datasets/v1/opcua/synthetic")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        (root / f"opcua_{i:03d}.xml").write_text(make_nodeset(i))


if __name__ == "__main__":
    main()
