import urllib.request, json

r = urllib.request.urlopen('http://localhost:8001/api/client-profiles')
data = json.loads(r.read())
print(f'Profiles returned: {len(data)}')
for p in data:
    md_len = len(p.get('active_profile_md') or '')
    has_json = bool(p.get('active_profile_json'))
    print(f'  name={p["name"]!r}  md_chars={md_len}  json={has_json}  analyzed={p.get("total_sops_analyzed")}')
