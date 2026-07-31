import csv
import os
import rhinoscriptsyntax as rs

folder = r"D:\UT_IT\Conferenca\2026\Architecture\ProgramaticDesign\Computational_Form-Finding_of_Arches\testImages"

filename = "arch_run{N}_{loadtype}_g{g}_k{k}.csv".format(
    N=N,
    loadtype=loadtype, 
    g=round(g_val, 3),
    k=round(k_val, 3)
)
path = os.path.join(folder, filename)

with open(path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z"])
    for p in pts:
        pt = rs.coerce3dpoint(p)
        writer.writerow([pt.X, pt.Y, pt.Z])

a = "Done: " + str(len(pts)) + " points written to " + filename