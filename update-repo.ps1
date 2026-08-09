# LEGACY SIDE-EFFECTFUL WORKFLOW
# This script pulls, broadly stages, commits, pushes, changes among hard-coded
# checkouts, and opens an HPC SSH session. It is not the standard repository
# update path. Review every target and obtain explicit user confirmation before
# running it. Whether this script should be retired or redesigned is Decision required.
# Each checkout's origin is expected to have the QMUL and GitHub.com push URLs
# documented in README.md, so one `git push origin main` updates both servers.

cd "C:\Users\exy053\Documents\00-PhD-gitRepo"
git pull
git add .
git commit -m "(Local) Auto-commit."
git push origin main

cd "C:\Users\exy053\OneDrive - Queen Mary, University of London\Documents\Research\00-PhD-gitRepo"
git pull
git add .
git commit -m "(OneDrive) Auto-commit."
git push origin main

cd "Z:\00-PhD-gitRepo"
git pull
git add .
git commit -m "(Z: Drive) Auto-commit."
git push origin main

cd "C:\Users\exy053\Documents\00-PhD-gitRepo"
git pull
ssh -i C:\Users\exy053\.ssh\id_rsa exy053@login.hpc.qmul.ac.uk
