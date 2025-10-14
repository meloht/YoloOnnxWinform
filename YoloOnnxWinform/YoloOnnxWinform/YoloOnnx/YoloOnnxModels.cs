using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloOnnxWinform.YoloOnnx
{
    public record LabelModel
    {
        /// <summary>
        /// Label index
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// Name of the label.
        /// </summary>
        public string Name { get; init; } = default!;
    }

    public record Output(int BatchSize, int Elements, int Channels, int Width, int Height)
    {
        public static Output Classification(int[] dimensions)
            => new(dimensions[0], dimensions[1], 0, 0, 0);

        public static Output Detection(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], 0, 0);

        public static Output Segmentation(int[] dimensions)
            => new(dimensions[0], 0, dimensions[1], dimensions[2], dimensions[3]);

        public static Output Empty() => new(0, 0, 0, 0, 0);
    }
    public record Input(int BatchSize, int Channels, int Height, int Width)
    {
        public static Input Shape(int[] dimensions)
            => new(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
    }
}
