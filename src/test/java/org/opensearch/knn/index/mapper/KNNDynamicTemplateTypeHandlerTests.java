/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNNDynamicTemplateTypeHandlerTests extends KNNTestCase {

    private final KNNDynamicTemplateTypeHandler handler = new KNNDynamicTemplateTypeHandler();

    /** A supplier over a flat numeric array of the given length — get() yields a parser at START_ARRAY. */
    private FieldValueParserSupplier arraySupplier(int dimension) {
        String json = "[" + IntStream.range(0, dimension).mapToObj(i -> "0.1").collect(Collectors.joining(",")) + "]";
        return supplierOver(json);
    }

    /** A supplier over the given JSON value bytes; get() creates a parser positioned at the first token. */
    private FieldValueParserSupplier supplierOver(String json) {
        return new FieldValueParserSupplier(
            MediaTypeRegistry.JSON,
            NamedXContentRegistry.EMPTY,
            LoggingDeprecationHandler.INSTANCE,
            json.getBytes(java.nio.charset.StandardCharsets.UTF_8)
        );
    }

    /** A supplier whose get() throws — asserts the handler must not read the value for a complete config. */
    private FieldValueParserSupplier failingSupplier() {
        return FieldValueParserSupplier.withoutValue();
    }

    // --- Explicit knn intent (type: knn_vector) -------------------------------------------------

    public void testClaimsAndInjectsDimensionFromArrayLength() throws IOException {
        // type: knn_vector without dimension — claimed, dimension injected from array length, no threshold.
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(64)));
        assertEquals(64, config.get(KNNConstants.DIMENSION));
    }

    public void testDimensionPresentOpensNoParser() throws IOException {
        // Complete config — claimed with no read.
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 64);
        assertTrue(handler.adjustMappingConfig(config, failingSupplier()));
        assertEquals(64, config.get(KNNConstants.DIMENSION));
    }

    public void testModelIdPresentOpensNoParserAndInjectsNoDimension() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        assertTrue(handler.adjustMappingConfig(config, failingSupplier()));
        assertFalse("dimension must not be injected when a model supplies it", config.containsKey(KNNConstants.DIMENSION));
    }

    // --- Knn signal without an explicit type (dimension / model_id only) ------------------------

    public void testDimensionOnlyInjectsTypeAndClaims() throws IOException {
        // { dimension: 32 } with no type — explicit knn signal, so claimed with no read and no threshold.
        Map<String, Object> config = new HashMap<>();
        config.put(KNNConstants.DIMENSION, 32);
        assertTrue(handler.adjustMappingConfig(config, failingSupplier()));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(32, config.get(KNNConstants.DIMENSION));
    }

    public void testModelIdOnlyInjectsTypeAndClaims() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put(KNNConstants.MODEL_ID, "my-model");
        assertTrue(handler.adjustMappingConfig(config, failingSupplier()));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertFalse(config.containsKey(KNNConstants.DIMENSION));
    }

    // --- Empty block — length threshold decides -------------------------------------------------

    public void testEmptyBlockClaimsAtThreshold() throws IOException {
        Map<String, Object> config = new HashMap<>();
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(128)));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(128, config.get(KNNConstants.DIMENSION));
    }

    public void testEmptyBlockClaimsNonMultipleOfEightAboveThreshold() throws IOException {
        Map<String, Object> config = new HashMap<>();
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(300)));
        assertEquals(300, config.get(KNNConstants.DIMENSION));
    }

    public void testEmptyBlockDeclinesBelowThreshold() throws IOException {
        Map<String, Object> config = new HashMap<>();
        assertFalse(handler.adjustMappingConfig(config, arraySupplier(127)));
        assertFalse(config.containsKey("type"));
        assertFalse(config.containsKey(KNNConstants.DIMENSION));
    }

    public void testEmptyBlockDeclinesNonArray() throws IOException {
        Map<String, Object> config = new HashMap<>();
        assertFalse(handler.adjustMappingConfig(config, supplierOver("\"not-an-array\"")));
        assertFalse(config.containsKey("type"));
    }

    // --- Knn-specific param (no type) signals intent, claims with no threshold ------------------

    public void testDataTypeOnlyClaimsBelowThreshold() throws IOException {
        // { data_type: float } is a knn-specific param — claim with no threshold, inject type + dimension.
        Map<String, Object> config = new HashMap<>();
        config.put(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float");
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(16)));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(16, config.get(KNNConstants.DIMENSION));
    }

    public void testMethodOnlyClaimsBelowThreshold() throws IOException {
        // { method: {...} } is a knn-specific param — claim with no threshold.
        Map<String, Object> config = new HashMap<>();
        config.put(KNNConstants.KNN_METHOD, new HashMap<>());
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(10)));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(10, config.get(KNNConstants.DIMENSION));
    }

    // --- Generic params are intent-neutral: behave like an empty block (threshold decides) ------

    public void testGenericParamOnlyBelowThresholdDeclines() throws IOException {
        // { store: true } carries no vector intent — apply the threshold; 64 < 128 declines.
        Map<String, Object> config = new HashMap<>();
        config.put("store", true);
        assertFalse(handler.adjustMappingConfig(config, arraySupplier(64)));
        assertFalse(config.containsKey("type"));
    }

    public void testGenericParamOnlyAtThresholdClaims() throws IOException {
        // { store: true } + 128-element array — threshold met, claim and inject type + dimension.
        Map<String, Object> config = new HashMap<>();
        config.put("store", true);
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(128)));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertEquals(128, config.get(KNNConstants.DIMENSION));
    }

    // --- Not ours -------------------------------------------------------------------------------

    public void testDeclinesOtherExplicitType() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", "some_other_type");
        assertFalse(handler.adjustMappingConfig(config, failingSupplier()));
        assertEquals("some_other_type", config.get("type"));
    }

    public void testUnknownParamNoTypeDeclines() throws IOException {
        // A stray unknown param with no knn type/signal — not ours; decline so it falls through.
        Map<String, Object> config = new HashMap<>();
        config.put("bogus_param", "x");
        assertFalse(handler.adjustMappingConfig(config, arraySupplier(200)));
        assertFalse(config.containsKey("type"));
    }

    public void testExplicitTypeWithUnknownParamClaimsAndLeavesItForTypeParser() throws IOException {
        // Explicit type: knn_vector with a typo'd param — claim it (do NOT silently decline). The unknown
        // key stays in the config so KNNVectorFieldMapper.TypeParser reports a clear error downstream.
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put("dimesnion", 128); // deliberate typo
        assertTrue(handler.adjustMappingConfig(config, arraySupplier(200)));
        assertEquals(KNNVectorFieldMapper.CONTENT_TYPE, config.get("type"));
        assertTrue("typo'd key must remain for the TypeParser to reject", config.containsKey("dimesnion"));
        // dimension is inferred from the array because the real 'dimension' key is absent
        assertEquals(200, config.get(KNNConstants.DIMENSION));
    }

    // --- isConfigComplete -----------------------------------------------------------------------

    public void testIsConfigCompleteWithDimension() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 128);
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigCompleteWithModelId() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigIncompleteWithoutDimensionOrModel() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        assertFalse(handler.isConfigComplete(config));
    }
}
