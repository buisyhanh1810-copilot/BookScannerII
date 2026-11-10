package com.bookscan.app

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

enum class OutMode { GRAY, BW, COLOR }
enum class SplitMode { AUTO, ALWAYS, NEVER }

data class ScanOptions(
    val mode: OutMode = OutMode.GRAY,
    val split: SplitMode = SplitMode.AUTO,
    val autoCrop: Boolean = true,
    val deskew: Boolean = true,
    val maxSide: Int = 3000
)

/**
 * Pipeline: ảnh chụp 2 trang -> tìm viền sách -> nắn phẳng -> tìm rãnh gáy -> tách đôi
 * -> xoá bóng/nền -> kéo trắng-đen -> chỉnh nghiêng -> trang chuẩn.
 */
object BookScanner {

    fun process(src: Bitmap, opt: ScanOptions): List<Bitmap> {
        var img = bitmapToBgr(src)
        img = limit(img, opt.maxSide)

        var spread = img
        if (opt.autoCrop) {
            // thuật toán mới (kết hợp sáng/tối, vùng chữ, bóng gáy); lỗi thì dùng thuật toán cũ
            val q = (try { EdgeDetector.detectQuad(img) } catch (e: Throwable) { null }) ?: detectQuad(img)
            if (q != null) spread = warp(img, q)
        }

        var ratio = spread.cols().toDouble() / spread.rows()
        if (opt.split == SplitMode.ALWAYS && ratio < 1.0) {
            val r = Mat()
            Core.rotate(spread, r, Core.ROTATE_90_CLOCKWISE)
            spread = r
            ratio = spread.cols().toDouble() / spread.rows()
        }
        val doSplit = when (opt.split) {
            SplitMode.ALWAYS -> true
            SplitMode.NEVER -> false
            SplitMode.AUTO -> ratio > 1.15
        }

        val pages = ArrayList<Mat>()
        if (doSplit) {
            val gx = try {
                EdgeDetector.findCut(spread).x.coerceIn(1, spread.cols() - 1)
            } catch (e: Throwable) {
                findGutter(spread)
            }
            pages.add(spread.submat(0, spread.rows(), 0, gx))
            pages.add(spread.submat(0, spread.rows(), gx, spread.cols()))
        } else {
            pages.add(spread)
        }
        return pages.map { toBitmap(cleanPage(it, opt)) }
    }

    // ---------- tiện ích ----------
    private fun bitmapToBgr(b: Bitmap): Mat {
        val rgba = Mat()
        Utils.bitmapToMat(b, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    private fun toBitmap(m: Mat): Bitmap {
        val rgba = Mat()
        Imgproc.cvtColor(
            m, rgba,
            if (m.channels() == 1) Imgproc.COLOR_GRAY2RGBA else Imgproc.COLOR_BGR2RGBA
        )
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        rgba.release()
        return bmp
    }

    private fun limit(m: Mat, maxSide: Int): Mat {
        val big = max(m.cols(), m.rows())
        if (big <= maxSide) return m
        val s = maxSide.toDouble() / big
        val out = Mat()
        Imgproc.resize(m, out, Size(), s, s, Imgproc.INTER_AREA)
        m.release()
        return out
    }

    // ---------- 1. tìm viền sách ----------
    private fun detectQuad(bgr: Mat): Array<Point>? {
        val scale = 800.0 / max(bgr.cols(), bgr.rows())
        val small = Mat()
        Imgproc.resize(bgr, small, Size(), scale, scale, Imgproc.INTER_AREA)
        val gray = Mat()
        Imgproc.cvtColor(small, gray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.GaussianBlur(gray, gray, Size(7.0, 7.0), 0.0)
        val bin = Mat()
        Imgproc.threshold(gray, bin, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(15.0, 15.0))
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, k)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, k)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(bin, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        val imgArea = small.cols().toDouble() * small.rows()
        val center = Point(small.cols() / 2.0, small.rows() / 2.0)
        var best: MatOfPoint? = null
        var bestScore = 0.0
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 0.2 * imgArea) continue
            val inside = Imgproc.pointPolygonTest(MatOfPoint2f(*c.toArray()), center, false) >= 0
            val score = area * (if (inside) 1.0 else 0.3)
            if (score > bestScore) { bestScore = score; best = c }
        }
        if (best == null) return null

        val pts = best.toArray()
        val hullIdx = MatOfInt()
        Imgproc.convexHull(best, hullIdx)
        val hull = hullIdx.toArray().map { pts[it] }.toTypedArray()
        val hull2f = MatOfPoint2f(*hull)
        val peri = Imgproc.arcLength(hull2f, true)

        var quad: Array<Point>? = null
        var eps = 0.01
        while (eps <= 0.12) {
            val ap = MatOfPoint2f()
            Imgproc.approxPolyDP(hull2f, ap, eps * peri, true)
            if (ap.rows() == 4) { quad = ap.toArray(); break }
            if (ap.rows() < 4) break
            eps += 0.005
        }
        if (quad == null) {
            val r = Imgproc.minAreaRect(hull2f)
            val p = Array(4) { Point() }
            r.points(p)
            quad = p
        }
        val scaled = quad.map { Point(it.x / scale, it.y / scale) }.toTypedArray()
        return orderPoints(scaled)
    }

    /** trả về [tl, tr, br, bl] */
    private fun orderPoints(p: Array<Point>): Array<Point> {
        val tl = p.minByOrNull { it.x + it.y }!!
        val br = p.maxByOrNull { it.x + it.y }!!
        val tr = p.minByOrNull { it.y - it.x }!!
        val bl = p.maxByOrNull { it.y - it.x }!!
        return arrayOf(tl, tr, br, bl)
    }

    private fun warp(bgr: Mat, q: Array<Point>): Mat {
        val tl = q[0]; val tr = q[1]; val br = q[2]; val bl = q[3]
        val w = max(hypot(tr.x - tl.x, tr.y - tl.y), hypot(br.x - bl.x, br.y - bl.y)).toInt().coerceAtLeast(200)
        val h = max(hypot(bl.x - tl.x, bl.y - tl.y), hypot(br.x - tr.x, br.y - tr.y)).toInt().coerceAtLeast(200)
        val src = MatOfPoint2f(tl, tr, br, bl)
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(w - 1.0, 0.0), Point(w - 1.0, h - 1.0), Point(0.0, h - 1.0)
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(bgr, out, m, Size(w.toDouble(), h.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE)
        return out
    }

    // ---------- 2. rãnh gáy ----------
    private fun findGutter(spread: Mat): Int {
        val w = spread.cols()
        val gray = Mat()
        Imgproc.cvtColor(spread, gray, Imgproc.COLOR_BGR2GRAY)
        val col = Mat()
        Core.reduce(gray, col, 0, Core.REDUCE_AVG, CvType.CV_32F)
        var k = w / 40
        if (k % 2 == 0) k++
        Imgproc.GaussianBlur(col, col, Size(max(k, 5).toDouble(), 1.0), 0.0)
        val a = FloatArray(w)
        col.get(0, 0, a)
        val lo = (w * 0.42).toInt()
        val hi = (w * 0.58).toInt()
        var bi = w / 2
        var bv = Float.MAX_VALUE
        for (x in lo..hi) if (a[x] < bv) { bv = a[x]; bi = x }
        val median = a.sorted()[w / 2]
        gray.release(); col.release()
        return if (bv > 0.97f * median) w / 2 else bi
    }

    // ---------- 3. làm sạch trang ----------
    private fun cleanPage(page: Mat, opt: ScanOptions): Mat {
        val ix = (page.cols() * 0.015).toInt()
        val iy = (page.rows() * 0.015).toInt()
        val roi = page.submat(iy, page.rows() - iy, ix, page.cols() - ix).clone()

        val color = opt.mode == OutMode.COLOR
        val flat: Mat
        if (color) {
            val chs = ArrayList<Mat>()
            Core.split(roi, chs)
            val f = chs.map { flatten(it) }
            flat = Mat()
            Core.merge(f, flat)
        } else {
            val g = Mat()
            Imgproc.cvtColor(roi, g, Imgproc.COLOR_BGR2GRAY)
            flat = flatten(g)
        }
        roi.release()

        val grayForLevels = if (color) {
            val g = Mat(); Imgproc.cvtColor(flat, g, Imgproc.COLOR_BGR2GRAY); g
        } else flat
        val lo = min(percentile(grayForLevels, 0.01), 110)
        val hi = 235
        val lut = Mat(1, 256, CvType.CV_8U)
        val tab = ByteArray(256) { i ->
            val t = ((i - lo).toDouble() / (hi - lo)).coerceIn(0.0, 1.0)
            (255 * t).toInt().toByte()
        }
        lut.put(0, 0, tab)
        var res = Mat()
        Core.LUT(flat, lut, res)

        if (opt.mode == OutMode.BW) {
            val bw = Mat()
            Imgproc.threshold(res, bw, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
            res = bw
        }

        if (opt.deskew) {
            val ang = estimateSkew(res)
            if (abs(ang) > 0.15) res = rotate(res, ang)
        }
        return res
    }

    /** chia ảnh cho nền (ước lượng bằng dilate) để xoá bóng và làm trắng giấy */
    private fun flatten(ch: Mat): Mat {
        val s = 0.25
        val small = Mat()
        Imgproc.resize(ch, small, Size(), s, s, Imgproc.INTER_AREA)
        var ks = max(small.cols(), small.rows()) / 18
        if (ks % 2 == 0) ks++
        ks = max(ks, 9)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(ks.toDouble(), ks.toDouble()))
        val bgS = Mat()
        Imgproc.dilate(small, bgS, kernel)
        Imgproc.GaussianBlur(bgS, bgS, Size(ks.toDouble(), ks.toDouble()), 0.0)
        val bg = Mat()
        Imgproc.resize(bgS, bg, ch.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val out = Mat()
        Core.divide(ch, bg, out, 255.0)
        small.release(); bgS.release(); bg.release()
        return out
    }

    private fun percentile(g: Mat, p: Double): Int {
        val s = Mat()
        Imgproc.resize(g, s, Size(256.0, 256.0), 0.0, 0.0, Imgproc.INTER_AREA)
        val b = ByteArray(256 * 256)
        s.get(0, 0, b)
        val h = IntArray(256)
        for (v in b) h[v.toInt() and 0xFF]++
        val target = (b.size * p).toInt()
        var acc = 0
        for (i in 0..255) { acc += h[i]; if (acc >= target) return i }
        return 0
    }

    // ---------- 4. chỉnh nghiêng ----------
    private fun estimateSkew(img: Mat): Double {
        val g = if (img.channels() == 1) img else {
            val t = Mat(); Imgproc.cvtColor(img, t, Imgproc.COLOR_BGR2GRAY); t
        }
        val sc = 500.0 / g.cols()
        val small = Mat()
        Imgproc.resize(g, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val mask = Mat()
        Imgproc.threshold(small, mask, 0.0, 255.0, Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_OTSU)
        val mx = (mask.cols() * 0.06).toInt()
        val my = (mask.rows() * 0.06).toInt()
        val inner = mask.submat(my, mask.rows() - my, mx, mask.cols() - mx).clone()
        val center = Point(inner.cols() / 2.0, inner.rows() / 2.0)
        var best = 0.0
        var bestScore = -1.0
        val rows = Mat()
        val rot = Mat()
        for (a in -40..40) {
            val angle = a * 0.125
            val m = Imgproc.getRotationMatrix2D(center, angle, 1.0)
            Imgproc.warpAffine(inner, rot, m, inner.size(), Imgproc.INTER_LINEAR)
            Core.reduce(rot, rows, 1, Core.REDUCE_SUM, CvType.CV_32F)
            val arr = FloatArray(rows.rows())
            rows.get(0, 0, arr)
            var score = 0.0
            for (i in 1 until arr.size) { val d = (arr[i] - arr[i - 1]).toDouble(); score += d * d }
            if (score > bestScore) { bestScore = score; best = angle }
        }
        return best
    }

    private fun rotate(img: Mat, angle: Double): Mat {
        val m = Imgproc.getRotationMatrix2D(Point(img.cols() / 2.0, img.rows() / 2.0), angle, 1.0)
        val out = Mat()
        Imgproc.warpAffine(img, out, m, img.size(), Imgproc.INTER_CUBIC,
            Core.BORDER_CONSTANT, Scalar(255.0, 255.0, 255.0))
        return out
    }
}
