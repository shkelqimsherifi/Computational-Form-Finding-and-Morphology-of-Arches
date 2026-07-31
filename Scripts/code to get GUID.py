doc = ghenv.Component.OnPingDocument()

if run:
    lines = []
    for obj in doc.Objects:
        nickname = obj.NickName
        guid = str(obj.InstanceGuid)
        lines.append("{0}  |  {1}".format(nickname, guid))
    a = "\n".join(sorted(lines))
else:
    a = "Set run=True to list components"