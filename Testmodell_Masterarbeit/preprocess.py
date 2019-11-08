from data.access import DataPipeLine
import os

# [1, 12, 55, 80, 17, 77, 71, 6, 28]
detail_wgs = [4891, 2426, 2363, 2456, 2447, 2453, 2451, 2448, 2455, 2454, 2764, 2452, 2444, 2443, 2427,
              2428, 2449, 2399, 2457, 2437, 2419, 2436, 2450]
dic = dict.fromkeys(detail_wgs)
dic2 = dict.fromkeys(detail_wgs)
pipeline = DataPipeLine(ZielWarengruppen=[55])
simulation_data = pipeline.get_regression_data()
print([df.shape for df in simulation_data])

for detail_wg in detail_wgs:
    print('Starte mit Warengruppe', detail_wg)
    simulation_params = {
        'ZielWarengruppen': [55],
        'DetailWarengruppe': [detail_wg]
    }
    try:
        pipeline = DataPipeLine(**simulation_params)
        simulation_data = pipeline.get_regression_data()
        dic[detail_wg] = [df.shape for df in simulation_data]

    except ValueError:  # Nach Filtern keine Daten mehr übrig. Erzeugt ValueError bei max(Absatzjahre)
        print('\t\tDas hat nicht geklappt')
