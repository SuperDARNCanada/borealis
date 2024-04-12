import sys
import scd_utils

if __name__ == "__main__":
    filename = sys.argv[1]

    scd_util = scd_utils.SCDUtils(filename)

    scd_util.add_line("20190404", "10:43", "twofsound", 'common')
    #scd_util.add_line("04/04/2019", "10:43", "twofsound")
    scd_util.add_line("20190407", "10:43", "twofsound", 'common')
    scd_util.add_line("20190414", "10:43", "twofsound", 'common')
    scd_util.add_line("20190414", "10:43", "twofsound", 'common', prio=1, duration='89')
    #scd_util.add_line("20190414", "10:43", "twofsound", prio=1, duration=24)
    scd_util.add_line("20190414", "11:43", "twofsound", 'common', duration='46')
    scd_util.add_line("20190414", "00:43", "twofsound", 'common')
    scd_util.add_line("20190408", "15:43", "twofsound", 'common', duration='57')

    scd_util.remove_line("20190414", "10:43", "twofsound", 'common')

    for line in scd_util.get_relevant_lines("20190414", "10:44"):
        print(line)

