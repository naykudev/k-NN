/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.index.mapper.DynamicValueSummary;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.Map;

public class KNNDynamicFieldTypeInferencerTests extends KNNTestCase {

    private final KNNDynamicFieldTypeInferencer inferencer = new KNNDynamicFieldTypeInferencer();

    /** The inferencer reads the summary; the supplier is unused, so a no-value one is fine. */
    private static final FieldValueParserSupplier NO_VALUE = FieldValueParserSupplier.withoutValue();

    public void testClaimsFlatNumericArrayAtThreshold() throws IOException {
        Map<String, Object> config = inferencer.inferFieldType(DynamicValueSummary.flatNumericArray(128), NO_VALUE);
        assertNotNull(config);
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(128, config.get("dimension"));
    }

    public void testClaimsMultipleOfEightAboveThreshold() throws IOException {
        Map<String, Object> config = inferencer.inferFieldType(DynamicValueSummary.flatNumericArray(256), NO_VALUE);
        assertNotNull("256-dim flat numeric array must be inferred as knn_vector", config);
        assertEquals(256, config.get("dimension"));
    }

    public void testNonMultipleOfEightNotClaimed() throws IOException {
        // 130 and 300 are >= 128 but not multiples of 8 — must NOT be claimed (%8 gate).
        assertNull("130 is not a multiple of 8", inferencer.inferFieldType(DynamicValueSummary.flatNumericArray(130), NO_VALUE));
        assertNull("300 is not a multiple of 8", inferencer.inferFieldType(DynamicValueSummary.flatNumericArray(300), NO_VALUE));
    }

    public void testBelowThresholdNotClaimed() throws IOException {
        // 120 is a multiple of 8 but below the minimum dimension.
        assertNull(inferencer.inferFieldType(DynamicValueSummary.flatNumericArray(120), NO_VALUE));
    }

    public void testNonNumericArrayNotClaimed() throws IOException {
        // Core classified it as a non-numeric array (e.g. contains a string) — decline regardless of length.
        assertNull(inferencer.inferFieldType(DynamicValueSummary.nonNumericArray(256), NO_VALUE));
    }

    public void testNonArrayNotClaimed() throws IOException {
        assertNull(inferencer.inferFieldType(DynamicValueSummary.scalar(), NO_VALUE));
        assertNull(inferencer.inferFieldType(DynamicValueSummary.object(), NO_VALUE));
    }
}
