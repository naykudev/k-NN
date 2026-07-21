/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNNDynamicFieldTypeInferencerTests extends KNNTestCase {

    private final KNNDynamicFieldTypeInferencer inferencer = new KNNDynamicFieldTypeInferencer();

    /** A supplier over a flat numeric array of the given length. */
    private FieldValueParserSupplier numericArray(int length) {
        String json = "[" + IntStream.range(0, length).mapToObj(i -> "0.1").collect(Collectors.joining(",")) + "]";
        return supplierOver(json);
    }

    private FieldValueParserSupplier supplierOver(String json) {
        return new FieldValueParserSupplier(
            MediaTypeRegistry.JSON,
            NamedXContentRegistry.EMPTY,
            LoggingDeprecationHandler.INSTANCE,
            json.getBytes(StandardCharsets.UTF_8)
        );
    }

    public void testClaimsArrayAtThreshold() throws IOException {
        Map<String, Object> config = inferencer.inferFieldType(numericArray(128));
        assertNotNull(config);
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(128, config.get("dimension"));
    }

    public void testClaimsNonMultipleOfEightDimension() throws IOException {
        // 130 and 300 are >= 128 but not multiples of 8 — must still be claimed (no %8 gate).
        Map<String, Object> config130 = inferencer.inferFieldType(numericArray(130));
        assertNotNull("130-dim array must be inferred as knn_vector", config130);
        assertEquals(130, config130.get("dimension"));

        Map<String, Object> config300 = inferencer.inferFieldType(numericArray(300));
        assertNotNull("300-dim array must be inferred as knn_vector", config300);
        assertEquals(300, config300.get("dimension"));
    }

    public void testBelowThresholdNotClaimed() throws IOException {
        assertNull(inferencer.inferFieldType(numericArray(127)));
    }

    public void testNonNumericArrayNotClaimed() throws IOException {
        // 128 elements but one is a string → not a flat numeric array.
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 127; i++) {
            sb.append("0.1,");
        }
        sb.append("\"x\"]");
        assertNull(inferencer.inferFieldType(supplierOver(sb.toString())));
    }

    public void testNonArrayNotClaimed() throws IOException {
        assertNull(inferencer.inferFieldType(supplierOver("\"not-an-array\"")));
    }
}
