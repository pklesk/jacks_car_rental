import matplotlib.pyplot as plt

if __name__ == "__main__":    
    sizes = [3_382_071, 6_456_681, 87_710_931, 167_448_141, 619_682_591, 1_183_030_401]    
    device_names = ["i7-10700@2.90GHz", "RTX 3090", "RTX 5090 Blackwell"]    
    series = [
        {"jcr_pi_contraction_cpu_numpy": [1.865, 2.549, 19.850, 22.218, 88.908, 98.928]},
        {        
            "jcr_pi_contraction_cuda_atomicmax": [0.029, 0.038, 0.085, 0.089, 0.520, 0.538],
            "jcr_pi_contraction_cuda_atomicmaxplain": [0.028, 0.038, 0.085, 0.089, 0.508, 0.527],
            "jcr_pi_contraction_cuda_reducemax": [0.022, 0.030, 0.082, 0.087, 0.517, 0.536],
            "jcr_pi_contraction_cuda_gridsync": [0.007, 0.010, 0.073, 0.078, 0.585, 0.597] 
        },
        {
            "jcr_pi_contraction_cuda_atomicmax": [0.010, 0.014, 0.032, 0.034, 0.164, 0.169],
            "jcr_pi_contraction_cuda_atomicmaxplain": [0.011, 0.015, 0.039, 0.043, 0.165, 0.171],
            "jcr_pi_contraction_cuda_reducemax": [0.008, 0.011, 0.031, 0.034, 0.158, 0.164],
            "jcr_pi_contraction_cuda_gridsync": [0.006, 0.007, 0.031, 0.033, 0.156, 0.160]            
        }
    ]

    method_colors = {
        "atomicmax": "#007FFF",          
        "atomicmaxplain": "#000080",   
        "reducemax": "#00C957",                 
        "gridsync": "#FF0000"            
    }
    title_str = "JACK'S CAR RENTAL: LOG-LOG PLOT OF SPEED-UPS"
    fig = plt.figure(figsize=(16, 5))
    try:
        fig.canvas.manager.set_window_title(title_str)
    except AttributeError:
        pass

    markers = ["o", "s", "^", "*"]
    last_x = sizes[-1]
    first_x = sizes[0]
    
    LABEL_FONT_SIZE = 8.5      
    LEGEND_FONT_SIZE = 10.5    
    AXIS_LABEL_SIZE = 15.0     
    TITLE_SIZE = 20.0          
    TICK_LABEL_SIZE = 10.0     

    COLUMN_SPACING = 1.45

    for i, device_dict in enumerate(series):
        device_name = device_names[i]
        methods = list(device_dict.items())
        
        line_style = "-" if "5090" in device_name else (0, (4, 2))
        
        for j, (method_name, timings) in enumerate(methods):
            full_label = f"[{device_name}] {method_name}"
            clean_name = method_name.replace("jcr_pi_contraction_cuda_", "")
            speed_ups = [cpu / gpu for cpu, gpu in zip(list(series[0].values())[0], timings)]
            
            if i == 0:
                plt.plot(sizes, speed_ups, marker="o", color="black", label=full_label,
                         linewidth=1.2, markersize=4, markerfacecolor="black", zorder=10)
                plt.text(last_x * 1.03, speed_ups[-1] * 1.05, "1.0x", color="black", 
                         fontsize=LABEL_FONT_SIZE, va="bottom", ha="left")
                continue

            color = method_colors.get(clean_name, "#7f7f7f")
            plt.plot(sizes, speed_ups, marker=markers[j % len(markers)], markerfacecolor="none",
                     markeredgewidth=1.5, color=color, label=full_label, 
                     linestyle=line_style, linewidth=1.8, markersize=7, zorder=10)

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
    
    plt.ylim(0.85, 800) 
    plt.xlim(first_x * 0.85, last_x * (COLUMN_SPACING ** 4.6)) 

    plt.xticks(sizes)
    
    x_labels = []
    for s in sizes:
        exponent = len(str(s)) - 1
        base = s / (10**exponent)
        x_labels.append(rf"${base:.1f} \cdot 10^{{{exponent}}}$")
    
    plt.gca().set_xticklabels(x_labels)

    plt.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    plt.tick_params(axis="y", which="minor", labelsize=TICK_LABEL_SIZE * 0.8)
    
    plt.xlabel("PROBLEM SIZE (ENTRIES IN JOINT DISTRIBUTION)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("SPEED-UP VS. CPU", fontsize=AXIS_LABEL_SIZE)
    plt.title(title_str, fontsize=TITLE_SIZE, pad=15)
    
    plt.grid(True, which="both", ls="-", color="#D3D3D3", alpha=0.3, zorder=1)
    
    plt.hlines(y=1, xmin=first_x, xmax=last_x, color="black", linestyle="-", 
               linewidth=1, alpha=0.2, zorder=2)
    
    plt.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE, framealpha=0.9, 
               ncol=1, handlelength=3.0, bbox_to_anchor=(0.99, 0.12), labelspacing=0.2)
    
    plt.tight_layout(pad=1.0)
    plt.show()