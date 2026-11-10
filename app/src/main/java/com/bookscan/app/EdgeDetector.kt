package com.bookscan.app

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
import java.util.Random
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * Phát hiện viền sách và rãnh gáy bằng cách KẾT HỢP nhiều đặc trưng tuyến tính:
 *
 *  1. Độ sáng/màu giấy so với nền: mô hình giấy học từ vùng giữa ảnh (độ sáng + sắc độ Lab),
 *     đo bước nhảy "giống giấy -> không giống giấy" ở nhiều tỉ lệ làm mờ (cạnh nét lẫn cạnh mờ).
 *  2. Vùng chữ: khối chữ phải nằm TRONG trang. Cạnh trang bị chặn không được cắt vào chữ
 *     (cổng text) và có tiên nghiệm lề giấy nằm cách chữ một khoảng hợp lý.
 *  3. Gáy sách: bóng gáy (cực tiểu độ sáng) + khoảng trống giữa hai khối chữ + vị trí gần giữa.
 *     Vết cắt luôn bị kẹp vào khoảng trống không chữ, cách chữ mỗi bên một lề an toàn,
 *     để không lẫn nội dung trang bên kia khi vùng giáp ranh bị mờ.
 *
 *  Mỗi cạnh được dò bằng nhiều tia vuông góc, mỗi tia cho một ứng viên + độ tin cậy,
 *  rồi khớp đường thẳng bền vững (RANSAC có trọng số) nên đoạn bị mờ/cong gần gáy tự bị loại.
 */
object EdgeDetector {

    class Cut(val x: Int, val risky: Boolean)

    private class V(val x: Double, val y: Double) {
        operator fun plus(o: V) = V(x + o.x, y + o.y)
        operator fun minus(o: V) = V(x - o.x, y - o.y)
        operator fun times(k: Double) = V(x * k, y * k)
        operator fun unaryMinus() = V(-x, -y)
        infix fun dot(o: V) = x * o.x + y * o.y
        fun len() = hypot(x, y)
    }

    private class Side(val mu: V, val dir: V, val ok: Boolean, val o: V)

    // ------------------------------------------------------------------ tiện ích
    private fun odd(n: Double): Int {
        val i = n.toInt()
        return if (i % 2 == 1) i else i + 1
    }

    private fun floats(m: Mat): FloatArray {
        val a = FloatArray(m.rows() * m.cols())
        m.get(0, 0, a)
        return a
    }

    private fun percentile(a: FloatArray, p: Double): Double {
        val s = a.sortedArray()
        val idx = p * (s.size - 1)
        val lo = floor(idx).toInt()
        val hi = ceil(idx).toInt()
        return s[lo] + (s[hi] - s[lo]) * (idx - lo)
    }

    private fun median(a: DoubleArray): Double {
        val s = a.sortedArray()
        return s[s.size / 2]
    }

    private fun clip(v: Double, lo: Double, hi: Double) = if (v < lo) lo else if (v > hi) hi else v

    private fun bil(a: FloatArray, w: Int, h: Int, x0: Double, y0: Double): Double {
        val x = clip(x0, 0.0, w - 1.001)
        val y = clip(y0, 0.0, h - 1.001)
        val xi = x.toInt()
        val yi = y.toInt()
        val fx = x - xi
        val fy = y - yi
        val i = yi * w + xi
        return a[i] * (1 - fx) * (1 - fy) + a[i + 1] * fx * (1 - fy) +
            a[i + w] * (1 - fx) * fy + a[i + w + 1] * fx * fy
    }

    private fun orderV(p: List<V>): Array<V> {
        val tl = p.minByOrNull { it.x + it.y }!!
        val br = p.maxByOrNull { it.x + it.y }!!
        val tr = p.minByOrNull { it.y - it.x }!!
        val bl = p.maxByOrNull { it.y - it.x }!!
        return arrayOf(tl, tr, br, bl)
    }

    private fun area(q: Array<V>): Double {
        var s = 0.0
        for (i in q.indices) {
            val a = q[i]
            val b = q[(i + 1) % q.size]
            s += a.x * b.y - b.x * a.y
        }
        return abs(s) / 2.0
    }

    private fun isConvex(q: Array<V>): Boolean {
        var sign = 0
        for (i in q.indices) {
            val a = q[i]
            val b = q[(i + 1) % 4]
            val c = q[(i + 2) % 4]
            val cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cr > 0) 1 else if (cr < 0) -1 else 0
            if (s == 0) return false
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return true
    }

    private fun rectQuad(pts: List<V>): Array<V> {
        val m = MatOfPoint2f(*pts.map { Point(it.x, it.y) }.toTypedArray())
        val r = Imgproc.minAreaRect(m)
        val p = Array(4) { Point() }
        r.points(p)
        return orderV(p.map { V(it.x, it.y) })
    }

    // ------------------------------------------------------------------ cue: vùng chữ
    /**
     * Mặt nạ nét chữ (0/1). Blackhat + ngưỡng có trễ (hysteresis): nét yếu/mờ chỉ giữ
     * nếu nằm gần nét mạnh. Vùng giữa [cx0,cx1) (gần gáy) dùng ngưỡng yếu thấp hơn
     * để bắt cả chữ bị mờ sát gáy.
     */
    private fun textMask(gray: Mat, cx0: Int = -1, cx1: Int = -1): Mat {
        val h = gray.rows()
        val w = gray.cols()
        val k = odd(max(9.0, 0.025 * max(h, w)))
        val kern = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(k.toDouble(), k.toDouble()))
        val bh = Mat()
        Imgproc.morphologyEx(gray, bh, Imgproc.MORPH_BLACKHAT, kern)
        val tmp = Mat()
        var thr = Imgproc.threshold(bh, tmp, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
        thr = max(thr, 14.0)
        val strong = Mat()
        Imgproc.threshold(bh, strong, thr, 1.0, Imgproc.THRESH_BINARY)
        val weak = Mat()
        Imgproc.threshold(bh, weak, max(8.0, 0.5 * thr), 1.0, Imgproc.THRESH_BINARY)
        if (cx0 >= 0 && cx1 > cx0) {
            val wk2 = Mat()
            Imgproc.threshold(bh, wk2, max(6.0, 0.35 * thr), 1.0, Imgproc.THRESH_BINARY)
            wk2.submat(0, h, cx0, cx1).copyTo(weak.submat(0, h, cx0, cx1))
            wk2.release()
        }
        val near = Mat()
        Imgproc.dilate(strong, near, Mat.ones(15, 15, CvType.CV_8U))
        val out = Mat()
        Core.bitwise_and(weak, near, out)
        bh.release(); tmp.release(); strong.release(); weak.release(); near.release()
        return out
    }

    // ------------------------------------------------------------------ 1. viền sách
    /** Trả về [tl, tr, br, bl] theo toạ độ ảnh gốc, hoặc null nếu không tìm được. */
    fun detectQuad(bgr: Mat): Array<Point>? {
        val sc = 700.0 / max(bgr.cols(), bgr.rows())
        val small = Mat()
        Imgproc.resize(bgr, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val w = small.cols()
        val h = small.rows()
        val long = max(w, h).toDouble()

        // --- Lab: L (đóng để xoá nét chữ), a, b ---
        val lab = Mat()
        Imgproc.cvtColor(small, lab, Imgproc.COLOR_BGR2Lab)
        val chs = ArrayList<Mat>()
        Core.split(lab, chs)
        val L8 = chs[0]
        val Lf = Mat()
        L8.convertTo(Lf, CvType.CV_32F)
        val Af = Mat()
        chs[1].convertTo(Af, CvType.CV_32F, 1.0, -128.0)
        val Bf = Mat()
        chs[2].convertTo(Bf, CvType.CV_32F, 1.0, -128.0)
        val kc = odd(max(9.0, 0.02 * long))
        val Lc = Mat()
        Imgproc.morphologyEx(
            Lf, Lc, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kc.toDouble(), kc.toDouble()))
        )
        Imgproc.GaussianBlur(Lc, Lc, Size(0.0, 0.0), 1.0)
        val lArr = floats(Lc)
        val aArr = floats(Af)
        val bArr = floats(Bf)

        // --- mô hình giấy từ vùng giữa ảnh ---
        val ry0 = (0.3 * h).toInt(); val ry1 = (0.7 * h).toInt()
        val rx0 = (0.3 * w).toInt(); val rx1 = (0.7 * w).toInt()
        val roiL = ArrayList<Float>()
        for (y in ry0 until ry1) for (x in rx0 until rx1) roiL.add(lArr[y * w + x])
        val roiArr = roiL.toFloatArray()
        val P = percentile(roiArr, 0.85)
        val thr60 = percentile(roiArr, 0.60)
        val selA = ArrayList<Float>()
        val selB = ArrayList<Float>()
        for (y in ry0 until ry1) for (x in rx0 until rx1) {
            if (lArr[y * w + x] >= thr60) { selA.add(aArr[y * w + x]); selB.add(bArr[y * w + x]) }
        }
        val ma = median(DoubleArray(selA.size) { selA[it].toDouble() })
        val mb = median(DoubleArray(selB.size) { selB[it].toDouble() })

        // --- bản đồ "giống giấy" Pm ---
        val chromArr = FloatArray(w * h)
        val brightArr = FloatArray(w * h)
        for (i in 0 until w * h) {
            brightArr[i] = clip((lArr[i] - 0.30 * P) / (0.30 * P), 0.0, 1.0).toFloat()
            val d = hypot(aArr[i] - ma, bArr[i] - mb)
            chromArr[i] = exp(-(d / 12.0) * (d / 12.0)).toFloat()
        }
        val chromM = Mat(h, w, CvType.CV_32F)
        chromM.put(0, 0, chromArr)
        Imgproc.GaussianBlur(chromM, chromM, Size(0.0, 0.0), 2.0)
        val chromB = floats(chromM)
        val pm = FloatArray(w * h) { brightArr[it] * chromB[it] }

        // --- vùng thô: Pm > 0.5, nối hai nửa giấy bị bóng gáy chia cắt ---
        val maskBytes = ByteArray(w * h) { if (pm[it] > 0.5f) 255.toByte() else 0 }
        var m = Mat(h, w, CvType.CV_8U)
        m.put(0, 0, maskBytes)
        val k = odd(0.02 * long)
        val rk = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(k.toDouble(), k.toDouble()))
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_CLOSE, rk)
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_OPEN, rk)
        val kw = odd(0.16 * long)
        Imgproc.morphologyEx(m, m, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(kw.toDouble(), 3.0)))
        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(m, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        var best: MatOfPoint? = null
        var bestScore = 0.0
        for (c in contours) {
            val a = Imgproc.contourArea(c)
            if (a < 0.12 * h * w) continue
            val inside = Imgproc.pointPolygonTest(MatOfPoint2f(*c.toArray()), Point(w / 2.0, h / 2.0), false) >= 0
            val s = a * (if (inside) 1.0 else 0.3)
            if (s > bestScore) { bestScore = s; best = c }
        }
        val r0 = Mat.zeros(h, w, CvType.CV_8U)
        val bestC = best
        if (bestC != null) Imgproc.drawContours(r0, listOf(bestC), -1, Scalar(255.0), -1)
        val full = bestC == null || Core.countNonZero(r0).toDouble() / (w * h) > 0.93

        // --- cue chữ: điểm nét chữ nằm trong các khối chữ đủ lớn ---
        var tm = textMask(L8)
        if (!full) {
            val er = Mat()
            val ek = odd(0.02 * long)
            Imgproc.erode(r0, er, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(ek.toDouble(), ek.toDouble())))
            val erB = Mat()
            Core.compare(er, Scalar(0.0), erB, Core.CMP_GT)
            val t2 = Mat()
            Core.bitwise_and(tm, erB, t2)
            tm = t2
        }
        val blocks = Mat()
        Imgproc.dilate(tm, blocks, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(odd(0.03 * w).toDouble(), odd(0.015 * h).toDouble())))
        val labels = Mat()
        val stats = Mat()
        val cents = Mat()
        val nCc = Imgproc.connectedComponentsWithStats(blocks, labels, stats, cents)
        val keep = BooleanArray(nCc)
        for (i in 1 until nCc) keep[i] = stats.get(i, Imgproc.CC_STAT_AREA)[0] >= 0.004 * h * w
        val labArr = IntArray(w * h)
        labels.get(0, 0, labArr)
        val tmBytes = ByteArray(w * h)
        tm.get(0, 0, tmBytes)
        val txs = ArrayList<Double>()
        val tys = ArrayList<Double>()
        for (i in 0 until w * h) {
            if (tmBytes[i].toInt() != 0 && keep[labArr[i]]) {
                txs.add((i % w).toDouble()); tys.add((i / w).toDouble())
            }
        }
        val stride = max(1, txs.size / 4000)
        val tp = ArrayList<V>()
        var ii = 0
        while (ii < txs.size) { tp.add(V(txs[ii], tys[ii])); ii += stride }
        val hasText = tp.size > 50

        // --- tứ giác thô ---
        val rough: Array<V>
        if (!full && bestC != null) {
            val pts = bestC.toArray()
            val hullIdx = MatOfInt()
            Imgproc.convexHull(bestC, hullIdx)
            val hullPts = hullIdx.toArray().map { pts[it] }.toTypedArray()
            val hull2f = MatOfPoint2f(*hullPts)
            val peri = Imgproc.arcLength(hull2f, true)
            var quad: Array<V>? = null
            var eps = 0.01
            while (eps <= 0.12) {
                val ap = MatOfPoint2f()
                Imgproc.approxPolyDP(hull2f, ap, eps * peri, true)
                if (ap.rows() == 4) { quad = ap.toArray().map { V(it.x, it.y) }.toTypedArray(); break }
                if (ap.rows() < 4) break
                eps += 0.005
            }
            rough = orderV((quad ?: rectQuad(hullPts.map { V(it.x, it.y) })).toList())
        } else if (hasText) {
            val q = rectQuad(tp)
            val c = V(q.sumOf { it.x } / 4, q.sumOf { it.y } / 4)
            rough = orderV(q.map { c + (it - c) * 1.18 })
        } else {
            val mg = 0.03
            rough = arrayOf(V(mg * w, mg * h), V((1 - mg) * w, mg * h), V((1 - mg) * w, (1 - mg) * h), V(mg * w, (1 - mg) * h))
        }
        val roughArea = area(rough)
        val cen = V(rough.sumOf { it.x } / 4, rough.sumOf { it.y } / 4)

        // --- dò từng cạnh bằng các tia vuông góc, kết hợp 3 cue ---
        val lsMat = Mat()
        Imgproc.GaussianBlur(Lc, lsMat, Size(0.0, 0.0), 0.8)
        val ls = floats(lsMat)
        val band = (0.12 * long).toInt()
        val nLines = 28
        val scales = intArrayOf(2, 4, 8, 14)
        val gam = doubleArrayOf(1.0, 0.97, 0.93, 0.88)
        val n = 2 * band + 1
        val far = 25
        val sides = ArrayList<Side>()

        for (i in 0 until 4) {
            val a = rough[i]
            val b = rough[(i + 1) % 4]
            val sl = (b - a).len()
            val dvec = (b - a) * (1.0 / sl)
            var o = V(dvec.y, -dvec.x)
            if (o dot ((a + b) * 0.5 - cen) < 0) o = -o

            var tmax = -1e9
            if (hasText) {
                val proj = FloatArray(tp.size) { (tp[it] dot o).toFloat() }
                tmax = percentile(proj, 0.997)
            }
            val marg = 0.006 * long

            val candP = ArrayList<V>()
            val candC = ArrayList<Double>()
            for (kk in 0 until nLines) {
                val t = 0.08 + 0.84 * kk / (nLines - 1)
                val c = a + (b - a) * t
                val pmv = DoubleArray(n)
                val lv = DoubleArray(n)
                val inb = BooleanArray(n)
                val pos = DoubleArray(n)
                for (j in 0 until n) {
                    val u = (j - band).toDouble()
                    val px = c.x + u * o.x
                    val py = c.y + u * o.y
                    pmv[j] = bil(pm, w, h, px, py)
                    lv[j] = bil(ls, w, h, px, py)
                    inb[j] = px >= 0 && px <= w - 1 && py >= 0 && py <= h - 1
                    pos[j] = px * o.x + py * o.y
                }
                val cp = DoubleArray(n + 1)
                val cl = DoubleArray(n + 1)
                for (j in 0 until n) { cp[j + 1] = cp[j] + pmv[j]; cl[j + 1] = cl[j] + lv[j] }
                val bs = DoubleArray(n)
                val lq = DoubleArray(n)
                for (si in scales.indices) {
                    val s = scales[si]
                    val g = gam[si]
                    for (j in s until n - s) {
                        val pin = (cp[j] - cp[j - s]) / s
                        val pout = (cp[j + s + 1] - cp[j + 1]) / s
                        bs[j] = max(bs[j], g * clip(pin - pout, 0.0, 1.0))
                        val lin = (cl[j] - cl[j - s]) / s
                        val lout = (cl[j + s + 1] - cl[j + 1]) / s
                        val dl = (lin - lout) / (0.30 * P)
                        lq[j] = max(lq[j], g * (clip(dl, 0.0, 1.0) + 0.5 * clip(-dl, 0.0, 1.0)))
                    }
                }
                val sArr = DoubleArray(n)
                for (j in 0 until n) {
                    val cons = if (j >= far) {
                        0.4 + 0.6 * clip(((cp[j] - cp[j - far]) / far - 0.3) / 0.5, 0.0, 1.0)
                    } else 0.4
                    val gate = clip((pos[j] - tmax) / max(marg, 1.0), 0.05, 1.0)
                    val prior = if (hasText) {
                        val z = (pos[j] - (tmax + 0.05 * long)) / (0.07 * long)
                        exp(-0.5 * z * z)
                    } else 0.0
                    sArr[j] = if (!inb[j] || j < max(far, 15) || j >= n - 15) 0.0
                    else gate * ((0.5 * bs[j] + 0.35 * lq[j]) * cons + 0.12 * prior)
                }
                var jb = 0
                for (j in 1 until n) if (sArr[j] > sArr[jb]) jb = j
                val smax = sArr[jb]
                var ex = -1.0
                for (j in 0 until n) if (abs(j - jb) > 8 && sArr[j] > ex) ex = sArr[j]
                val s2 = max(ex, median(sArr))
                val conf = clip((smax - s2) / 0.35, 0.0, 1.0)
                var off = 0.0
                if (jb > 1 && jb < n - 2) {
                    val den = sArr[jb - 1] - 2 * sArr[jb] + sArr[jb + 1]
                    if (abs(den) > 1e-6) off = clip(0.5 * (sArr[jb - 1] - sArr[jb + 1]) / den, -1.0, 1.0)
                }
                val u = (jb - band) + off
                candP.add(c + o * u)
                candC.add(conf)
            }

            // RANSAC có trọng số độ tin cậy
            val tol = 0.006 * long
            val rnd = Random((7 + i).toLong())
            val valid = candC.indices.filter { candC[it] > 0.15 }
            var bestIn: BooleanArray? = null
            var bestSc = 0.0
            if (valid.size >= 4) {
                for (iter in 0 until 100) {
                    val p = valid[rnd.nextInt(valid.size)]
                    var q = valid[rnd.nextInt(valid.size)]
                    if (q == p) continue
                    var dv = candP[q] - candP[p]
                    val ln = dv.len()
                    if (ln < 0.2 * sl) continue
                    dv = dv * (1.0 / ln)
                    val nv = V(-dv.y, dv.x)
                    val inl = BooleanArray(candP.size)
                    var sc2 = 0.0
                    for (idx in candP.indices) {
                        val dist = abs((candP[idx] - candP[p]) dot nv)
                        if (dist < tol && candC[idx] > 0.15) { inl[idx] = true; sc2 += candC[idx] }
                    }
                    if (sc2 > bestSc) { bestSc = sc2; bestIn = inl }
                }
            }
            val bi = bestIn
            if (bi != null && bi.count { it } >= 4) {
                var sw = 0.0; var mx = 0.0; var my = 0.0
                for (idx in candP.indices) if (bi[idx]) {
                    sw += candC[idx]; mx += candC[idx] * candP[idx].x; my += candC[idx] * candP[idx].y
                }
                mx /= sw; my /= sw
                var sxx = 0.0; var sxy = 0.0; var syy = 0.0
                for (idx in candP.indices) if (bi[idx]) {
                    val dx = candP[idx].x - mx
                    val dy = candP[idx].y - my
                    sxx += candC[idx] * dx * dx; sxy += candC[idx] * dx * dy; syy += candC[idx] * dy * dy
                }
                val ang = 0.5 * atan2(2 * sxy, sxx - syy)
                sides.add(Side(V(mx, my), V(cos(ang), sin(ang)), true, o))
            } else {
                sides.add(Side((a + b) * 0.5, dvec, false, o))
            }
        }

        // --- ràng buộc: mọi nét chữ phải nằm trong tứ giác (đẩy cạnh ra ngoài nếu cần) ---
        val adj = sides.map { sd ->
            var nv = V(sd.dir.y, -sd.dir.x)
            if ((nv dot sd.o) < 0) nv = -nv
            var mu = sd.mu
            if (hasText) {
                var v = -1e18
                for (t in tp) { val d = (t - mu) dot nv; if (d > v) v = d }
                val need = v + 0.006 * long
                if (need > 0) mu = mu + nv * need
            }
            Side(mu, sd.dir, sd.ok, sd.o)
        }

        fun inter(l1: Side, l2: Side): V? {
            val det = l1.dir.x * (-l2.dir.y) - (-l2.dir.x) * l1.dir.y
            if (abs(det) < 1e-6) return null
            val r = l2.mu - l1.mu
            val t = (r.x * (-l2.dir.y) - (-l2.dir.x) * r.y) / det
            return l1.mu + l1.dir * t
        }

        val corners = (0 until 4).map { inter(adj[(it + 3) % 4], adj[it]) }
        var result: Array<V> = rough
        if (corners.all { it != null }) {
            val q = orderV(corners.map { it!! })
            val ar = area(q)
            val okInside = q.all { it.x > -0.05 * w && it.x < 1.05 * w && it.y > -0.05 * h && it.y < 1.05 * h }
            if (isConvex(q) && okInside && ar > 0.5 * roughArea && ar < 1.6 * roughArea) result = q
        }
        val fin = result.map {
            Point(clip(it.x, 0.0, w - 1.0) / sc, clip(it.y, 0.0, h - 1.0) / sc)
        }
        return fin.toTypedArray()
    }

    // ------------------------------------------------------------------ 2. rãnh gáy
    /**
     * Vị trí cắt đôi trên ảnh đã nắn phẳng. Kết hợp: bóng gáy + khoảng trống giữa hai khối chữ
     * + gần giữa. Điểm cắt luôn bị kẹp vào khoảng không có chữ (cách chữ mỗi bên một lề an toàn).
     * risky = true khi chữ lấn sát gáy, không có khoảng trống an toàn.
     */
    fun findCut(spread: Mat): Cut {
        val sc = 700.0 / spread.cols()
        val small = Mat()
        Imgproc.resize(spread, small, Size(), sc, sc, Imgproc.INTER_AREA)
        val w = small.cols()
        val h = small.rows()
        val g = Mat()
        Imgproc.cvtColor(small, g, Imgproc.COLOR_BGR2GRAY)
        val g32 = Mat()
        g.convertTo(g32, CvType.CV_32F)
        val kc = odd(max(9.0, 0.025 * max(h, w)))
        val lc = Mat()
        Imgproc.morphologyEx(
            g32, lc, Imgproc.MORPH_CLOSE,
            Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(kc.toDouble(), kc.toDouble()))
        )
        val x0 = (0.3 * w).toInt()
        val x1 = (0.7 * w).toInt()
        val tm = textMask(g, x0, x1)
        val mx = (0.03 * w).toInt()
        val my = (0.03 * h).toInt()
        val zero = Scalar(0.0)
        tm.submat(0, my, 0, w).setTo(zero)
        tm.submat(h - my, h, 0, w).setTo(zero)
        tm.submat(0, h, 0, mx).setTo(zero)
        tm.submat(0, h, w - mx, w).setTo(zero)

        // bóng gáy: độ tối trung bình theo cột
        val col = Mat()
        Core.reduce(lc.submat((0.1 * h).toInt(), (0.9 * h).toInt(), 0, w), col, 0, Core.REDUCE_AVG, CvType.CV_32F)
        val kk = odd(max(5.0, w / 40.0))
        Imgproc.GaussianBlur(col, col, Size(kk.toDouble(), 1.0), 0.0)
        val cm = floats(col)
        val ref = percentile(cm.copyOfRange((0.1 * w).toInt(), (0.9 * w).toInt()), 0.85)
        val dn = DoubleArray(w) { clip(clip(1 - cm[it] / ref, 0.0, 1.0) / 0.15, 0.0, 1.0) }

        // khoảng trống chữ: khoảng cách tới cột có chữ gần nhất
        val colSum = Mat()
        Core.reduce(tm, colSum, 0, Core.REDUCE_SUM, CvType.CV_32F)
        val cs = floats(colSum)
        val raw = BooleanArray(w) { cs[it] / h > 0.004 }
        val tc = BooleanArray(w) { x -> raw[x] || (x > 0 && raw[x - 1]) || (x < w - 1 && raw[x + 1]) }
        val dist = DoubleArray(w)
        var last = -1e9
        for (x in 0 until w) { if (tc[x]) last = x.toDouble(); dist[x] = x - last }
        last = 1e9
        for (x in w - 1 downTo 0) { if (tc[x]) last = x.toDouble(); dist[x] = min(dist[x], last - x) }

        var xb = w / 2
        var bv = -1e9
        for (x in x0 until x1) {
            val gap = clip(dist[x] / (0.025 * w), 0.0, 1.0)
            val z = (x - w / 2.0) / (0.08 * w)
            val prior = exp(-0.5 * z * z)
            val s = 0.40 * dn[x] + 0.40 * gap + 0.20 * prior
            if (s > bv) { bv = s; xb = x }
        }
        val margin = max(2, (0.008 * w).toInt())
        var xl = xb
        while (xl > 0 && !tc[xl]) xl--
        var xr = xb
        while (xr < w - 1 && !tc[xr]) xr++
        val risky: Boolean
        val xf: Int
        if (xr - xl >= 2 * margin + 2) {
            xf = xb.coerceIn(xl + margin, xr - margin)
            risky = false
        } else {
            xf = xb
            risky = true
        }
        val xFull = Math.round(xf / sc).toInt().coerceIn(1, spread.cols() - 1)
        return Cut(xFull, risky)
    }
}
