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

  // 最终重绘期间页面会调用 onReset；此时不得中止仍在读取的流，
  // 否则服务端随后补发的 finish 行会被丢弃。
  var rendering = false;

  // 把致命错误交给页面现有的错误展示逻辑。
  function emitFatal(msg) {
    if (typeof window.MeetingUI.handleEvent === "function") {
      window.MeetingUI.handleEvent({ t: "error", msg: msg, fatal: true });
    }
  }

  // 把服务端 turn 层字段适配成页面原有的单句事件形状。
  function adaptEvent(event) {
    if (!event) {
      return event;
    }
    if (event.turn_id) {
      event.utt_id = event.turn_id;
    }
    if (event.t === "asr_partial" && typeof event.turn_text === "string") {
      event.text = event.turn_text;
    }
    if (event.t === "asr_final" && typeof event.turn_text === "string") {
      event.raw_text = event.turn_text;
    }
    if (event.t === "llm_confirmed" && typeof event.turn_polished_text === "string") {
      event.polished_text = event.turn_polished_text;
    }
    return event;
  }

  // 用最终 turns 重绘整篇记录；turns 为空时保留流式视图。
  function renderTurns(event) {
    var ui = window.MeetingUI;
    rendering = true;
    try {
      if (typeof ui.reset === "function") {
        ui.reset();
      }
      var mapping = {};
      for (var i = 0; i < event.turns.length; i += 1) {
        var turn = event.turns[i];
        ui.handleEvent({
          t: "asr_final",
          utt_id: turn.turn_id,
          start_ms: turn.start_ms,
          end_ms: turn.end_ms,
          raw_text: turn.text
        });
        ui.handleEvent({
          t: "llm_confirmed",
          utt_id: turn.turn_id,
          start_ms: turn.start_ms,
          end_ms: turn.end_ms,
          raw_text: turn.text,
          polished_text: turn.text,
          corrected: false,
          bypassed: false,
          error: null
        });
        mapping[turn.turn_id] = {
          spk: turn.spk,
          spk_label: turn.spk_label,
          overlap_spk: null
        };
      }
      ui.handleEvent({
        t: "speaker_diarized",
        n_clusters: event.n_clusters,
        mapping: mapping
      });
      ui.handleEvent({ t: "finish" });
    } finally {
      rendering = false;
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
      if (event.t === "speaker_diarized" &&
          Array.isArray(event.turns) && event.turns.length > 0) {
        renderTurns(event);
        return;
      }
      window.MeetingUI.handleEvent(adaptEvent(event));
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
      if (rendering) {
        return;
      }
      if (controller) {
        controller.abort();
        controller = null;
      }
    };
  }
})();
