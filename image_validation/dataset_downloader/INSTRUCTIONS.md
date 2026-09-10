# 1. Inspect dataset status per row
python main.py --diagnostic
# 2. Perform a dry-run check without downloading
python main.py --dry-run
# 3. Perform a quick smoke test on the first 3 pending rows
python main.py --test
# 4. Download a specific Excel row (e.g. row 102)
python main.py --test-row 102
# 5. Run using installed Helium browser
python main.py --helium --test-row 102
# 6. Run the full dataset download loop
python main.py