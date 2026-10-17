/* 会议回放桥接：把静态原型页的 window.MeetingUI 接到回放服务。
 * 只补齐 onAudioSimulation / onReset 两个钩子，不触碰麦克风相关钩子。
 * 无依赖；语法保持 ES5（var / function），与页面内联脚本风格一致。
 */
(function () {
  "use strict";

  // 页面尚未暴露 MeetingUI（脚本未加载或加载顺序变化）时直接退出。
  if (!window.MeetingUI) {
    return;
  }

  // 已有其它脚本接入音频模拟时保持原实现，绝不覆盖。
  if (typeof window.MeetingUI.onAudioSimulation === "function") {
    return;
  }

  // 模块级唯一的在途请求控制器；重置或再次发起时终止旧请求。
  var controller = null;

  // 把致命错误交给页面现有的错误展示逻辑。
  function emitFatal(msg) {
    if (typeof window.MeetingUI.handleEvent === "function") {
      window.MeetingUI.handleEvent({ t: "error", msg: msg, fatal: true });
    }
  }

  // 处理一行完整的 NDJSON；空行与单行 JSON 解析失败都直接忽略。
  function dispatchLine(line) {
    var text = line.replace(/\r$/, "");
    if (!text) {
      return;
    }
    var event;
    try {
      event = JSON.parse(text);
    } catch (err) {
      return;
    }
    if (typeof window.MeetingUI.handleEvent === "function") {
      window.MeetingUI.handleEvent(event);
    }
  }

  // 逐块读取 NDJSON 流：按换行切分，未闭合的尾段留在 buffer 中。
  function pump(reader, decoder, buffer) {
    return reader.read().then(function (result) {
      if (result.done) {
        // 流结束时缓冲区可能还剩下最后一行（没有结尾换行），补发一次。
        dispatchLine(buffer + decoder.decode());
        return;
      }
      var text = decoder.decode(result.value, { stream: true });
      var parts = (buffer + text).split("\n");
      buffer = parts.pop();
      for (var i = 0; i < parts.length; i += 1) {
        dispatchLine(parts[i]);
      }
      return pump(reader, decoder, buffer);
    });
  }

  // 音频文件模拟：以 multipart 上传到回放服务，并消费 NDJSON 事件流。
  window.MeetingUI.onAudioSimulation = function (file) {
    // 同一时刻只保留一个在途请求。
    if (controller) {
      controller.abort();
    }
    controller = new AbortController();
    var current = controller;

    // 页面查询参数 fast=1 表示加速回放；缺省按实时节奏（fast=0）。
    var fast = "0";
    try {
      var requested = new URLSearchParams(location.search).get("fast");
      if (requested) {
        fast = requested;
      }
    } catch (err) {
      fast = "0";
    }

    var form = new FormData();
    form.append("file", file, file.name);

    var url = "/api/stream_meeting?name=" + encodeURIComponent(file.name) +
              "&fast=" + encodeURIComponent(fast);

    fetch(url, {
      method: "POST",
      body: form,
      signal: current.signal
    }).then(function (response) {
      if (!response.ok) {
        emitFatal("HTTP " + response.status);
        return;
      }
      if (!response.body || typeof response.body.getReader !== "function") {
        emitFatal("当前浏览器不支持流式响应");
        return;
      }
      return pump(response.body.getReader(), new TextDecoder("utf-8"), "");
    }).catch(function (err) {
      // 重置触发的主动终止不是错误，静默返回。
      if (err && err.name === "AbortError") {
        return;
      }
      emitFatal(String(err));
    });
  };

  // 重置：终止在途请求并清空控制器；其它脚本已提供 onReset 时保留原实现。
  if (typeof window.MeetingUI.onReset !== "function") {
    window.MeetingUI.onReset = function () {
      if (controller) {
        controller.abort();
        controller = null;
      }
    };
  }
})();
