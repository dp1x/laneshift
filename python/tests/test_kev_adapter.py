import asyncio
import json
import unittest

import httpx

from policies.model_adapter import KevIdentityError, KevInferenceError, KevResponseError, KevSystemOnePolicy


def sample_state():
    return {
        "protocol_version": 1,
        "episode_id": "ep-44-2",
        "scenario_seed": 44,
        "sim_time": 2.0,
        "physics_tick": 120,
        "state_id": "ep-44-2:120",
        "player": {
            "lane": 1,
            "target_lane": 1,
            "lane_change_in_progress": False,
            "speed": 24.0,
            "position": {"x": 0.0, "y": 0.0, "z": 48.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 24.0},
        },
        "traffic": [{
            "id": "veh_8",
            "lane": 1,
            "position": {"x": 0.0, "y": 0.0, "z": 60.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 19.0},
            "relative_distance_z": 12.0,
            "relative_speed_z": 5.0,
        }],
        "legal_actions": ["LEFT", "STAY", "BRAKE"],
    }


class KevAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_builds_typed_choice_request_and_parses_actual_response_shape(self):
        captured = []

        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "jaredpalmer/kev-0.8b", "base": "Qwen/Qwen3.5-0.8B-Base"}]})
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"x-typesafe-request-id": "req-abc"},
                json={
                    "model": "kev-latest",
                    "answers": {"next_action": {
                        "type": "choice",
                        "choice": "LEFT",
                        "confidence": 0.82,
                        "probabilities": {"LEFT": 0.84, "STAY": 0.11, "BRAKE": 0.05},
                    }},
                    "usage": {"input_tokens": 111, "output_tokens": 72},
                    "latency_ms": 42.0,
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test:8009",
                "model": "kev-latest",
                "expected_run": "jaredpalmer/kev-0.8b",
                "label_source": "original_model",
                "timeout": 0.4,
            })
            await policy.start()
            decision = await policy.choose_action(sample_state())

        request = captured[0]
        self.assertEqual(request["model"], "kev-latest")
        self.assertEqual(request["state"]["state_id"], "ep-44-2:120")
        question = request["questions"]["next_action"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]), {"LEFT", "STAY", "BRAKE"})
        self.assertNotIn("RIGHT", question["criteria"])
        self.assertEqual(decision.action, "LEFT")
        self.assertEqual(decision.label_source, "original_model")
        self.assertEqual(decision.model_id, "jaredpalmer/kev-0.8b")
        self.assertEqual(decision.probabilities, {"LEFT": 0.84, "STAY": 0.11, "BRAKE": 0.05})
        self.assertEqual(decision.model_latency_ms, 42.0)
        self.assertEqual(decision.request_id, "req-abc")
        self.assertEqual(decision.request_payload, request)

    async def test_refuses_to_label_a_different_loaded_checkpoint(self):
        async def handler(_request):
            return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "unexpected/run"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test",
                "model": "kev-latest",
                "expected_run": "jaredpalmer/kev-0.8b",
                "label_source": "original_model",
            })
            with self.assertRaises(KevIdentityError):
                await policy.start()

    async def test_rejects_model_choice_outside_the_offered_legal_actions(self):
        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "runs/laneshift-ft"}]})
            return httpx.Response(200, json={
                "model": "kev-latest",
                "answers": {"next_action": {
                    "type": "choice", "choice": "RIGHT",
                    "probabilities": {"LEFT": 0.1, "STAY": 0.2, "BRAKE": 0.1, "RIGHT": 0.6},
                }},
                "latency_ms": 10.0,
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test",
                "model": "kev-latest",
                "expected_run": "runs/laneshift-ft",
                "label_source": "fine_tuned_model",
            })
            await policy.start()
            with self.assertRaises(KevResponseError) as caught:
                await policy.choose_action(sample_state())
            self.assertEqual(caught.exception.details["response_payload"]["answers"]["next_action"]["choice"], "RIGHT")

    async def test_rejects_missing_choice_answer(self):
        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "runs/model"}]})
            return httpx.Response(200, json={"model": "kev-latest", "answers": {}, "latency_ms": 1.0})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test",
                "expected_run": "runs/model",
                "label_source": "original_model",
            })
            await policy.start()
            with self.assertRaises(KevResponseError):
                await policy.choose_action(sample_state())

    async def test_http_timeout_is_reported_as_inference_failure(self):
        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "runs/model"}]})
            raise httpx.ReadTimeout("mock inference timeout", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test",
                "expected_run": "runs/model",
                "label_source": "original_model",
                "timeout": 0.05,
            })
            await policy.start()
            with self.assertRaisesRegex(KevInferenceError, "Kev System One request failed") as caught:
                await policy.choose_action(sample_state())
            self.assertEqual(caught.exception.details["request_payload"]["state"]["episode_id"], "ep-44-2")
            self.assertIsNone(caught.exception.details["response_payload"])

    async def test_invalid_probability_distribution_is_rejected(self):
        async def handler(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"models": [{"name": "kev-latest", "run": "runs/model"}]})
            return httpx.Response(200, json={
                "model": "kev-latest",
                "answers": {"next_action": {
                    "type": "choice", "choice": "LEFT",
                    "probabilities": {"LEFT": 0.2, "STAY": 0.2, "BRAKE": 0.2},
                }},
                "latency_ms": 10.0,
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = KevSystemOnePolicy(client, {
                "base_url": "http://kev.test",
                "expected_run": "runs/model",
                "label_source": "original_model",
            })
            await policy.start()
            with self.assertRaises(KevResponseError) as caught:
                await policy.choose_action(sample_state())
            self.assertEqual(caught.exception.details["response_payload"]["answers"]["next_action"]["choice"], "LEFT")


if __name__ == "__main__":
    unittest.main()
