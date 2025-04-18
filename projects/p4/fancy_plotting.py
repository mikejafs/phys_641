import matplotlib as mpl

def fancy_plotting(use_tex):
    if use_tex:
        mpl.rcParams.update({
            # Enable LaTeX for all text rendering
            'text.usetex': True,

            # Set the font to Computer Modern (the default LaTeX font)
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],

            # Figure and font sizes
            'font.size': 11,             # Base font size
            'axes.labelsize': 14,        # Axis labels
            'axes.titlesize': 16,        # Axes titles
            'xtick.labelsize': 12,       # X tick labels
            'ytick.labelsize': 12,       # Y tick labels
            'legend.fontsize': 12,       # Legend text size

            # Figure properties
            'figure.figsize': [6.4, 4.8],  # Default figure size in inches
            'figure.dpi': 100,             # Default figure resolution

            # Save figures with high resolution
            'savefig.dpi': 300,

            # Make sure negative signs are rendered correctly
            'axes.unicode_minus': False,

            # Adjust the layout automatically to fit labels
            'figure.autolayout': True,

            # LaTeX preamble: load extra packages such as amsmath and amssymb
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
        })
