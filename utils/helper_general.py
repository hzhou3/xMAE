import yaml
from tabulate import tabulate

def print_cfg_as_tables(cfg, *, only=None, exclude=None, keys=None,
                        tablefmt="github", stralign="left"):
    """
    Pretty-print nested cfg (dict) using tabulate, preserving structure.

    Notes:
      • Paths are like "cfg", "cfg.data", "cfg.loss_weights".
      • Matching is prefix-based: selecting "cfg.data" includes its children.
      • Lists: scalar lists become one table; lists of dicts print each element as its own table.
    """
    def path_of(parent, child):
        return f"{parent}.{child}" if parent else str(child)

    def allowed(path):
        if only:
            if any(path.startswith(p) or p.startswith(path) for p in only):
                return True
            return False
        return True

    def excluded(path):
        return any(path.startswith(p) for p in (exclude or []))

    def fmt(v):
        if v is None: return "null"
        if isinstance(v, bool): return "true" if v else "false"
        if isinstance(v, float): return f"{v:g}"
        if isinstance(v, (list, tuple)) and all(not isinstance(x, (dict, list, tuple)) for x in v):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)

    def section_title(title):
        print(f"\n### {title}\n")

    def visit(obj, path):
        if excluded(path) or not allowed(path):
            return

        if isinstance(obj, dict):
            # rows for scalar leaves in this dict
            wh = (keys or {}).get(path)
            rows = []
            nested = []
            for k, v in obj.items():  # keep original order
                p = path_of(path, k)
                if isinstance(v, (dict, list, tuple)):
                    nested.append((k, v, p))
                else:
                    if (wh is None) or (k in wh):
                        rows.append([k, fmt(v)])

            if rows:
                section_title(path or "cfg")
                print(tabulate(rows, headers=[path or "cfg", "Value"],
                               tablefmt=tablefmt, stralign=stralign))

            # descend
            for k, v, p in nested:
                visit(v, p)

        elif isinstance(obj, (list, tuple)):
            if all(not isinstance(x, (dict, list, tuple)) for x in obj):
                rows = [[i, fmt(x)] for i, x in enumerate(obj)]
                section_title(path)
                print(tabulate(rows, headers=[f"{path}[i]", "Value"],
                               tablefmt=tablefmt, stralign=stralign))
            else:
                for i, x in enumerate(obj):
                    visit(x, f"{path}[{i}]")
        else:
            # scalar root (unlikely)
            section_title(path or "cfg")
            print(tabulate([[path or "cfg", fmt(obj)]],
                           headers=["Field", "Value"], tablefmt=tablefmt, stralign=stralign))

    visit(cfg, "cfg")




def read_from_cfg(args = None):
    if args is None:
        print('[ERROR] provide args...')
        exit()

    file_path = args.cfg
        
    """Reads a YAML file and returns its contents as a Python dictionary."""
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['cfg_path'] = args.cfg
    config['experiment'] = args.experiment
    
    print_cfg_as_tables(
        config,
        # only=["cfg", "cfg.data", "cfg.train"],
        exclude=['cfg.from_s3', 'cfg.data', 'cfg.train', 'cfg.callback'],
        keys={
            "cfg": ["model", "source", "dataset", "qual_thre"],
            "cfg.loss_weights": ["mwm_ppg", "mask_ratio_ppg", "mask_mode", 'mwm_ecg', 'contrast', 'consistency_ppg', 'fir_reg', 'clip'],

        }
    )

    return config


def save_to_cfg(data_dict, file_path: None):
    if file_path is None:
        print('[ERROR] provide where to save cfg path...')
        exit()
    """Saves a Python dictionary to a YAML file."""
    with open(file_path, 'w') as f:
        # default_flow_style=False makes it look like a standard YAML file
        yaml.dump(data_dict, f, default_flow_style=False)
