"""console summary and Table 1 figure for replicate_table1.py results"""

from __future__ import annotations

from families import FAMILIES

from self_play_pretrain_zero_data.executors.base import exists


def print_summary(results):
    print(f'\n{"family":<12}{"earliest round":>15}{"uniform prior E[round]":>24}   example\n')

    for family in FAMILIES:
        record, expected, seen = results['discoveries'].get(family), results['baseline']['expected_round'][family], results['baseline']['hits'][family]
        expected_str = '-' if not exists(expected) else (f'{expected:,.0f}' if seen else f'>{expected:,.0f}')
        example = f'{record["program"]!r} -> {record["output_head"][:8]}' if exists(record) else '-'
        print(f'{family:<12}{record["round"] if exists(record) else "not found":>15}{expected_str:>24}   {example}')

def make_plot(results, path):
    try:
        import matplotlib
    except ImportError:
        print('matplotlib is not installed, skipping the figure')
        return

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    discoveries, baseline = results['discoveries'], results['baseline']
    colors = plt.get_cmap('tab10').colors
    families = list(reversed(FAMILIES))

    fig, (ax_timeline, ax_loss) = plt.subplots(1, 2, figsize = (13, 5))

    for row, family in enumerate(families):
        record, expected = discoveries.get(family), baseline['expected_round'][family]

        if exists(record):
            ax_timeline.scatter(record['round'], row, s = 90, color = colors[0], zorder = 3)
        else:
            ax_timeline.text(0.02, row, 'not found', transform = ax_timeline.get_yaxis_transform(), color = colors[0], fontsize = 9, va = 'center')

        if not exists(expected):
            continue

        if baseline['hits'][family] == 0:
            ax_timeline.annotate('', xy = (expected, row), xytext = (expected * 1.6, row), arrowprops = dict(arrowstyle = '->', color = colors[1]))
            ax_timeline.text(expected * 1.8, row, f'>{expected:,.0f}', color = colors[1], fontsize = 9, va = 'center')
        else:
            ax_timeline.scatter(expected, row, s = 90, marker = 'D', color = colors[1], zorder = 3)

    legend = [Line2D([], [], marker = 'o', color = 'w', markerfacecolor = colors[0], markersize = 9, label = 'self-play observed'),
              Line2D([], [], marker = 'D', color = 'w', markerfacecolor = colors[1], markersize = 9, label = 'uniform prior expected')]

    ax_timeline.set_yticks(range(len(families)), labels = families)
    ax_timeline.set_xscale('log')
    ax_timeline.set_xlabel('round of first discovery')
    ax_timeline.set_title(f'families discovered during self-play\n(vs uniform prior at {results["config"]["detect_samples"]} programs / round)', fontsize = 10)
    ax_timeline.axvline(results.get('rounds_completed', len(results['losses'])), color = 'gray', linestyle = ':', linewidth = 1)
    ax_timeline.grid(axis = 'x', alpha = 0.3)
    ax_timeline.legend(handles = legend, loc = 'lower right', fontsize = 8)

    losses = results['losses']
    ax_loss.plot(range(1, len(losses) + 1), losses, color = colors[2], linewidth = 1)

    for family in FAMILIES:
        if exists(record := discoveries.get(family)):
            ax_loss.axvline(record['round'], alpha = 0.5, linewidth = 1)
            ax_loss.annotate(family, (record['round'], max(losses) * 0.98), rotation = 90, fontsize = 8, va = 'top', ha = 'right')

    ax_loss.set_xlabel('self-play round')
    ax_loss.set_ylabel('learner loss (nats / byte)')
    ax_loss.set_title('learner training loss', fontsize = 10)
    ax_loss.grid(alpha = 0.3)

    fig.suptitle('minimal replication of Table 1: recognizable math sequences from zero-data self-play')
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)
