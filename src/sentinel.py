"""
Purposed radar_status module. A daemon for controlling, monitoring, and managing borealis.

This module is named after the large horse that watches over you at the PGR radar site.
"""
import random

# console = Console()
# table = Table(show_header=True, header_style="bold magenta")

# table.add_column("Date", style="dim", width=12)
# table.add_column("Title")
# table.add_column("Production Budget", justify="right")
# table.add_column("Box Office", justify="right")
# table.add_row(
#     "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
# )
# table.add_row(
#     "May 25, 2018",
#     "[red]Solo[/red]: A Star Wars Story",
#     "$275,000,000",
#     "$393,151,347",
# )
# table.add_row(
#     "Dec 15, 2017",
#     "Star Wars Ep. VIII: The Last Jedi",
#     "$262,000,000",
#     "[bold]$1,332,539,889[/bold]",
# )
#
# console.print(table)
#
# for step in track(range(100)):
#     time.sleep(0.05)
#
# tasks = [f"task {n}" for n in range(1, 11)]
#
# with console.status("[bold green]Working on tasks...", spinner='pong') as status:
#     while tasks:
#         task = tasks.pop(0)
#         time.sleep(1)
#         console.log(f"{task} complete")

# import plotext as plt
# import numpy as np

# x = np.arange(1, 360+1, 1)
# y = np.sin(np.deg2rad(x))
# plt.subplot()
# plt.theme('pro')
# plt.plot_size(512, 256) # char dimensions
# while True:
#     plt.plot(x, y)
#     plt.ylim(-1, 1)
#     plt.show()
#     x[:-1] = x[1:]
#     y[:-1] = y[1:]
#     x[-1] = x[-2] + 1
#     y[-1] = np.sin(np.deg2rad(x[-1] % 360))
#     plt.clear_data()
#     plt.clear_terminal()

# path = 'steamed_hams.gif'
# plt.download('https://media.giphy.com/media/l2JeblO3xpDqTOAr6/giphy.gif', path)
# plt.plot_size(64, 64)
# plt.play_gif(path)
# plt.delete_file(path)


import time


def dumb_funk():
    log.error(f"Sample Func ERROR", key="value", another_key=[1, 2, 3])
    return


def main():
    dumb_funk()
    while True:
        log.info(f"loop", random_value=random.random())
        time.sleep(2)
        # Generate a crash
        with open('notafile.txt', 'r') as f:
            print('here')


if __name__ == '__main__':
    from utils import log_config
    log = log_config.log(log_level='INFO')
    log.warning(f"Booted", module="sentinel")
    try:
        main()
    except Exception as e:
        log.critical("main crashed", exception=e)



# def make_plot(width, height, phase=0, title=""):
#     plt.clf()
#     l, frames = 1000, 30
#     x = range(1, l + 1)
#     y = plt.sin(periods=2, length=l, phase=2 * phase/frames)
#     plt.scatter(x, y)
#     plt.plotsize(width, height)
#     plt.xaxes(1, 0)
#     plt.yaxes(1, 0)
#     plt.title(title)
#     plt.theme('pro')
#     plt.ylim(-1, 1)
#     return plt.build()
#
#
# class plotextMixin(JupyterMixin):
#     def __init__(self, phase=0, title=""):
#         self.decoder = AnsiDecoder()
#         self.phase = phase
#         self.title = title
#
#     def __rich_console__(self, console, options):
#         self.width = options.max_width or console.width
#         self.height = options.height or console.height
#         canvas = make_plot(self.width, self.height, self.phase, self.title)
#         self.rich_canvas = Group(*self.decoder.decode(canvas))
#         yield self.rich_canvas
#
#
# def make_layout():
#     layout = Layout(name="root")
#     layout.split(
#         Layout(name="header", size=5),
#         Layout(name="main", ratio=1),
#     )
#     layout["main"].split_column(
#         Layout(name="static", ratio=1),
#         Layout(name="dynamic"),
#     )
#     return layout
#
# print('asdadadasdasdas')
# layout = make_layout()
# print("jfjfjfgkfgkgjh")
#
# header = layout['header']
# title = plt.colorize("Pl✺text ", "cyan+", "bold") + "integration with " + plt.colorize("rich_", style="dim")
# header.update(Text(title, justify="center"))
#
# static = layout["static"]
# phase = 0
# mixin_static = Panel(plotextMixin(title="Static Plot"))
# static.update(mixin_static)
#
# dynamic = layout["dynamic"]
#
# with Live(layout, refresh_per_second=0.0001) as live:
#     while True:
#         phase += 1
#         mixin_dynamic = Panel(plotextMixin(phase, "Dynamic Plot"))
#         dynamic.update(mixin_dynamic)
#         live.refresh()


