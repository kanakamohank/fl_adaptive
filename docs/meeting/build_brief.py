import base64, pathlib
SP=pathlib.Path("/tmp/claude-0/-home-user-fl-adaptive/f013d0e7-2fd0-532d-bcdc-5ca152ee2779/scratchpad")
img=base64.b64encode(pathlib.Path("results/meeting/overview.png").read_bytes()).decode()
html=pathlib.Path(SP/"notes_body.html").read_text()
html=html.replace("{{FIGURE}}", f"data:image/png;base64,{img}")
pathlib.Path(SP/"tavs_status.html").write_text(html)
print("built", len(html), "chars")
