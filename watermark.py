"""Credit line stamped in the bottom-right corner of every generated image."""

WATERMARK = "Made with the LogP/TPSA Visualiser, developed by M. Benardout"


def add_watermark(fig) -> None:
    """Add the credit line below everything else in the figure, right-aligned.

    It sits just under the figure's lower edge; saving with bbox_inches="tight"
    grows the image to include it, so it never covers a plot or colour bar.
    """
    fig.text(0.99, -0.01, WATERMARK, ha="right", va="top",
             fontsize=7, color="#8a8a8a", style="italic")
