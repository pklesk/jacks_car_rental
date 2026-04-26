__author__ = "Przemysław Klęsk"
__email__ = "pklesk@zut.edu.pl"

import matplotlib.pyplot as plt

if __name__ == "__main__":
        
    sizes = [3_382_071, 6_456_681, 87_710_931, 167_448_141, 619_682_591, 1_183_030_401]  
    device_names = ["i7-10700", "Ryzen 9 9950X", "RTX 3090", "RTX 5090 Blackwell"]    
    series = [
        {
            "jcr_pi_contraction_cpu_numpy": [1.894, 2.544, 19.454, 21.900, 85.480, 87.424],
            "jcr_pi_contraction_cpu_numba_parallel": [0.028, 0.036, 0.446, 0.463, 3.764, 4.017],
        },
        {
            "jcr_pi_contraction_cpu_numpy": [1.054, 1.441, 10.541, 11.207, 44.117, 45.692],
            "jcr_pi_contraction_cpu_numba_parallel": [0.009, 0.011, 0.141, 0.177, 2.536, 2.744],
        },
        {        
            "jcr_pi_contraction_cuda_atomicmax": [0.028, 0.040, 0.086, 0.090, 0.521, 0.538],
            "jcr_pi_contraction_cuda_atomicmaxplain": [0.029, 0.040, 0.085, 0.089, 0.509, 0.526],
            "jcr_pi_contraction_cuda_reducemax": [0.022, 0.032, 0.082, 0.087, 0.518, 0.534],
            "jcr_pi_contraction_cuda_gridsync": [0.007, 0.010, 0.072, 0.077, 0.591, 0.594]
        },
        {
            "jcr_pi_contraction_cuda_atomicmax": [0.010, 0.014, 0.032, 0.034, 0.165, 0.170],
            "jcr_pi_contraction_cuda_atomicmaxplain": [0.011, 0.015, 0.039, 0.043, 0.165, 0.171],
            "jcr_pi_contraction_cuda_reducemax": [0.008, 0.011, 0.031, 0.034, 0.158, 0.164],
            "jcr_pi_contraction_cuda_gridsync": [0.006, 0.007, 0.031, 0.033, 0.156, 0.159]            
        }
    ]

    method_colors = {
        "jcr_pi_contraction_cpu_numba_parallel": "#FFA500",
        "jcr_pi_contraction_cuda_atomicmax": "#007FFF",          
        "jcr_pi_contraction_cuda_atomicmaxplain": "#000080",   
        "jcr_pi_contraction_cuda_reducemax": "#00C957",         
        "jcr_pi_contraction_cuda_gridsync": "#FF0000"            
    }

    title_str = "JACK'S CAR RENTAL: LOG-LOG PLOT OF SPEED-UPS"
    fig = plt.figure(figsize=(16, 7))
    try:
        fig.canvas.manager.set_window_title(title_str)
    except AttributeError:
        pass

    markers_cpu = [".", "."]
    markers_cuda = ["o", "s", "^", "v", "*"]
    last_x = sizes[-1]
    first_x = sizes[0]
    
    LABEL_FONT_SIZE = 8.5      
    LEGEND_FONT_SIZE = 10.5    
    AXIS_LABEL_SIZE = 15.0     
    TITLE_SIZE = 20.0          
    TICK_LABEL_SIZE = 12.0     
    COLUMN_SPACING = 1.45

    for i, device_dict in enumerate(series):
        device_name = device_names[i]
        methods = list(device_dict.items())
        
        line_style = "-" if ("5090" in device_name or "Ryzen" in device_name) else (0, (4, 2))
        
        markers = markers_cpu if i < 2 else markers_cuda                    
        for j, (method_name, timings) in enumerate(methods):
            full_label = f"[{device_name}] {method_name}"
            clean_name = method_name # intermediate mechanism (in case sth should be replaced form the method name, now inactive)
            speed_ups = [cpu / gpu for cpu, gpu in zip(list(series[0].values())[0], timings)]
            
            default_color = "black" if i < 2 else "#7f7f7f"
            color = method_colors.get(clean_name, default_color)
            current_lw = 1.7 if (i == 0 and j == 0) else 1.7
            current_ms = 7 if (i == 0 and j == 0) else 7
            
            if i == 0:                
                plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], color=color, label=full_label,
                         linewidth=current_lw, markersize=current_ms, markerfacecolor=color, 
                         linestyle=line_style, zorder=10)                                
                if j == 0:
                    plt.text(last_x * 1.03, speed_ups[-1] * 1.05, "1.0x", color="black", 
                             fontsize=LABEL_FONT_SIZE, va="bottom", ha="left")                
            else:
                plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], markerfacecolor="none",
                         markeredgewidth=1.5, color=color, label=full_label, 
                         linestyle=line_style, linewidth=current_lw, markersize=current_ms, zorder=10)
            if i == 0 and j == 0:
                continue
            
            last_y = speed_ups[-1]
            x_pos = last_x * (COLUMN_SPACING ** (j + 1)) 
            
            plt.hlines(y=last_y, xmin=last_x, xmax=x_pos, colors="black", 
                       linewidth=0.5, alpha=0.15, zorder=2)
            
            bbox_props = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
            plt.text(x_pos, last_y, f"{last_y:.1f}x", color=color, 
                     fontsize=LABEL_FONT_SIZE, va="center", ha="center", 
                     bbox=bbox_props, zorder=11)

    plt.xscale("log")
    plt.yscale("log")
    plt.ylim(0.85, 725.0) 
    
    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE * 0.8)
    
    plt.xlim(first_x * 0.85, last_x * (COLUMN_SPACING ** 5.6)) 
    
    plt.xlabel("PROBLEM SIZE (ENTRIES IN JOINT DISTRIBUTION)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("SPEED-UP VS. CPU", fontsize=AXIS_LABEL_SIZE)
    plt.title(title_str, fontsize=TITLE_SIZE, pad=15)
    
    plt.grid(True, which="both", ls="-", color="#F5F5F5", zorder=1)    
        
    plt.legend(loc="center left", bbox_to_anchor=(0.0, 0.35), fontsize=LEGEND_FONT_SIZE, framealpha=0.9, ncol=1, handlelength=3.0, labelspacing=0.2)
    
    plt.tight_layout(pad=1.0)
    plt.show()
