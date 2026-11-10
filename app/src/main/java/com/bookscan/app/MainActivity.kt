package com.bookscan.app

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var spMode: Spinner
    private lateinit var spSplit: Spinner
    private lateinit var cbCrop: CheckBox
    private lateinit var cbDeskew: CheckBox
    private lateinit var tvStatus: TextView
    private lateinit var progress: ProgressBar
    private lateinit var results: LinearLayout
    private lateinit var btnCamera: Button
    private lateinit var btnPick: Button
    private lateinit var btnPdf: Button

    private val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
    private var counter = 0
    private val saved = mutableListOf<Uri>()
    private var camFile: File? = null
    private val pendingFiles = mutableListOf<File>()

    // Chọn nhiều ảnh cùng lúc từ thư viện
    private val pickImages =
        registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
            if (uris.isNotEmpty()) processAll(uris)
        }

    // Chụp liên tục: mỗi lần chụp xong tự mở lại camera; thoát camera thì xử lý cả loạt
    private val takePicture =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
            val f = camFile
            if (ok && f != null && f.length() > 0) {
                pendingFiles.add(f)
                tvStatus.text = "Đã chụp ${pendingFiles.size} ảnh – chụp tiếp, hoặc thoát camera để xử lý"
                launchCamera()
            } else {
                f?.delete()
                camFile = null
                if (pendingFiles.isNotEmpty()) {
                    val batch = pendingFiles.toList()
                    pendingFiles.clear()
                    processAll(batch.map { fileUri(it) }, batch)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        savedInstanceState?.let { b ->
            b.getString("cam")?.let { camFile = File(it) }
            b.getStringArrayList("pending")?.forEach { pendingFiles.add(File(it)) }
        }
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "Không nạp được OpenCV", Toast.LENGTH_LONG).show()
        }

        spMode = findViewById(R.id.spMode)
        spSplit = findViewById(R.id.spSplit)
        cbCrop = findViewById(R.id.cbCrop)
        cbDeskew = findViewById(R.id.cbDeskew)
        tvStatus = findViewById(R.id.tvStatus)
        progress = findViewById(R.id.progress)
        results = findViewById(R.id.results)
        btnCamera = findViewById(R.id.btnCamera)
        btnPick = findViewById(R.id.btnPick)
        btnPdf = findViewById(R.id.btnPdf)

        spMode.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("Xám – xoá bóng (giống bản scan)", "Đen trắng", "Màu – xoá bóng")
        )
        spSplit.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            listOf("Tự động", "Luôn tách đôi", "Không tách (1 trang)")
        )

        spMode.setSelection(OutMode.BW.ordinal) // mặc định: đen trắng

        btnCamera.setOnClickListener { pendingFiles.clear(); launchCamera() }
        btnPick.setOnClickListener { pickImages.launch("image/*") }
        btnPdf.setOnClickListener { exportPdf() }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString("cam", camFile?.absolutePath)
        outState.putStringArrayList("pending", ArrayList(pendingFiles.map { it.absolutePath }))
    }

    private fun options() = ScanOptions(
        mode = OutMode.values()[spMode.selectedItemPosition],
        split = SplitMode.values()[spSplit.selectedItemPosition],
        autoCrop = cbCrop.isChecked,
        deskew = cbDeskew.isChecked
    )

    private fun launchCamera() {
        val dir = File(cacheDir, "images").apply { mkdirs() }
        val f = File(dir, "cap_${System.currentTimeMillis()}.jpg")
        camFile = f
        takePicture.launch(fileUri(f))
    }

    private fun fileUri(f: File): Uri =
        FileProvider.getUriForFile(this, "$packageName.fileprovider", f)

    private fun setBusy(b: Boolean) {
        progress.visibility = if (b) View.VISIBLE else View.GONE
        btnCamera.isEnabled = !b
        btnPick.isEnabled = !b
        btnPdf.isEnabled = !b
    }

    private fun processAll(uris: List<Uri>, tempFiles: List<File> = emptyList()) {
        val opt = options()
        setBusy(true)
        lifecycleScope.launch {
            var total = 0
            for ((i, uri) in uris.withIndex()) {
                tvStatus.text = "Đang xử lý ảnh ${i + 1}/${uris.size}…"
                try {
                    val outs = withContext(Dispatchers.Default) { processOne(uri, opt) }
                    for ((u, th) in outs) addThumb(u, th)
                    total += outs.size
                } catch (e: Throwable) {
                    tvStatus.text = "Lỗi ở ảnh ${i + 1}: ${e.message}"
                }
            }
            tempFiles.forEach { it.delete() }
            tvStatus.text = "Xong: $total trang từ ${uris.size} ảnh, đã lưu vào Pictures/BookScanner"
            setBusy(false)
        }
    }

    private fun processOne(uri: Uri, opt: ScanOptions): List<Pair<Uri, Bitmap>> {
        val bmp = Exporter.decode(this, uri)
        val pages = BookScanner.process(bmp, opt)
        bmp.recycle()
        return pages.map { p ->
            counter++
            val name = "scan_${stamp}_%03d".format(counter)
            val u = Exporter.saveJpeg(this, p, name)
            val th = Bitmap.createScaledBitmap(p, 500, (500f * p.height / p.width).toInt().coerceAtLeast(1), true)
            p.recycle()
            saved.add(u)
            u to th
        }
    }

    private fun addThumb(uri: Uri, th: Bitmap) {
        val iv = ImageView(this).apply {
            setImageBitmap(th)
            adjustViewBounds = true
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(uri, "image/jpeg")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION))
            }
        }
        results.addView(iv, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT))
    }

    private fun exportPdf() {
        if (saved.isEmpty()) {
            Toast.makeText(this, "Chưa có trang nào", Toast.LENGTH_SHORT).show()
            return
        }
        setBusy(true)
        tvStatus.text = "Đang tạo PDF…"
        lifecycleScope.launch {
            try {
                withContext(Dispatchers.IO) { Exporter.savePdf(this@MainActivity, saved.toList(), "scan_$stamp") }
                tvStatus.text = "Đã lưu PDF vào Download/BookScanner"
            } catch (e: Throwable) {
                tvStatus.text = "Lỗi PDF: ${e.message}"
            }
            setBusy(false)
        }
    }
}
