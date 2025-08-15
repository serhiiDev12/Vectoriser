using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using System.Globalization;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Drawing;

class Program
{
    static void Main(string[] args)
    {
        string inputFolder = @"D:/to-outline";
        string outputFolder = @"D:/vectorised";

        string[] files = Directory.GetFiles(inputFolder, "*.*");
        string inputImagePath = files.FirstOrDefault(f =>
            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase));

        if (inputImagePath == null)
        {
            Console.WriteLine("No input image found.");
            return;
        }

        Console.WriteLine($"Processing: {inputImagePath}");

        using var img = new Image<Bgr, byte>(inputImagePath);

        int width = img.Width;
        int height = img.Height;

        var uniqueColors = new HashSet<Color>();

        for (int y = 0; y < img.Height; y++)
        {
            for (int x = 0; x < img.Width; x++)
            {
                var pixel = img[y, x];
                var color = Color.FromArgb((int)pixel.Red, (int)pixel.Green, (int)pixel.Blue);
                uniqueColors.Add(color);
            }
        }

        Console.WriteLine($"Unique colors found: {uniqueColors.Count}");

        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>");

        foreach (var color in uniqueColors)
        {
            using var mask = new Image<Gray, byte>(img.Size);

            for (int y = 0; y < img.Height; y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    var pixel = img[y, x];
                    if (pixel.Red == color.R && pixel.Green == color.G && pixel.Blue == color.B)
                        mask[y, x] = new Gray(255);
                }
            }

            using var contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(mask, contours, null, Emgu.CV.CvEnum.RetrType.External, Emgu.CV.CvEnum.ChainApproxMethod.ChainApproxNone);

            for (int i = 0; i < contours.Size; i++)
            {
                var contour = contours[i];
                if (contour.Size < 4) continue; // Need at least 4 points for Catmull-Rom

                var points = contour.ToArray().ToList();

                sb.Append($"<path fill='rgb({color.R},{color.G},{color.B})' stroke='#1c1c1c' stroke-width='0.3' d='");

                // Start
                var p0 = points[0];
                sb.Append($"M {p0.X},{p0.Y} ");

                for (int j = 0; j < points.Count; j++)
                {
                    var p1 = points[j];
                    var p2 = points[(j + 1) % points.Count];
                    var p3 = points[(j + 2) % points.Count];
                    var p4 = points[(j + 3) % points.Count];

                    var b = CatmullRomToBezier(p1, p2, p3, p4);

                    sb.Append($"C {b[0].X},{b[0].Y} {b[1].X},{b[1].Y} {b[2].X},{b[2].Y} ");
                }

                sb.Append("Z' />\n");
            }
        }

        sb.AppendLine("</svg>");

        Directory.CreateDirectory(outputFolder);
        string outputSvg = Path.Combine(outputFolder, "vectorised.svg");
        File.WriteAllText(outputSvg, sb.ToString());

        Console.WriteLine($"✅ Smooth Bezier vector saved: {outputSvg}");
    }

    static PointF[] CatmullRomToBezier(Point p0, Point p1, Point p2, Point p3)
    {
        // Convert Catmull-Rom segment to cubic Bezier control points
        // Source: https://pomax.github.io/bezierinfo/#catmullconv

        var bp1 = new PointF(
            p1.X + (p2.X - p0.X) / 6f,
            p1.Y + (p2.Y - p0.Y) / 6f);

        var bp2 = new PointF(
            p2.X - (p3.X - p1.X) / 6f,
            p2.Y - (p3.Y - p1.Y) / 6f);

        var bp3 = new PointF(p2.X, p2.Y);

        return new[] { bp1, bp2, bp3 };
    }
}