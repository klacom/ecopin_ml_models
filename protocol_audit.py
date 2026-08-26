import sys, ast
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def audit_script(path, label):
    with open(path, encoding='utf-8', errors='replace') as f:
        src = f.read()
    tree = ast.parse(src)

    augs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'RandomHorizontalFlip':
                for kw in node.keywords:
                    if kw.arg == 'p': augs['H-flip p'] = kw.value.value
            if node.func.attr == 'RandomRotation':
                for kw in node.keywords:
                    if kw.arg == 'degrees': augs['Rotation degrees'] = kw.value.value

    consts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ('STAGE1_EPOCHS','BATCH_SIZE','LR_STAGE1','LR_STAGE2','PATIENCE','SEED','MAX_EPOCHS'):
                    if isinstance(node.value, ast.Constant):
                        consts[t.id] = node.value.value

    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    artifact = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and 'artifacts' in n.value]

    print(f'\n[{label}]')
    print(f'  Augmentations          : {augs}')
    print(f'  STAGE1_EPOCHS          : {consts.get("STAGE1_EPOCHS")}')
    print(f'  SEED                   : {consts.get("SEED")}')
    print(f'  BATCH_SIZE             : {consts.get("BATCH_SIZE")}')
    print(f'  LR_STAGE1              : {consts.get("LR_STAGE1")}')
    print(f'  LR_STAGE2              : {consts.get("LR_STAGE2")}')
    print(f'  PATIENCE               : {consts.get("PATIENCE")}')
    print(f'  MAX_EPOCHS             : {consts.get("MAX_EPOCHS")}')
    print(f'  EarlyStopping classes  : {classes}')
    print(f'  ARTIFACT_ROOT target   : {artifact[:1]}')
    has_test = 'TEST_CSV' in src
    has_cj   = 'ColorJitter' in src
    has_ls   = 'label_smoothing' in src
    has_dr   = 'drop_rate=' in src
    has_wd   = 'weight_decay=' in src
    print(f'  TEST_CSV referenced    : {has_test}')
    print(f'  ColorJitter            : {has_cj}')
    print(f'  label_smoothing        : {has_ls}')
    print(f'  drop_rate= (dropout)   : {has_dr}')
    print(f'  weight_decay=          : {has_wd}')

audit_script('training/train_corrected_baseline.py', 'Corrected Baseline')
audit_script('training/train_experiment3.py', 'Experiment 3')
